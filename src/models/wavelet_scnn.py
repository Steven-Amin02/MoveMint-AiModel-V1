import math
import torch
import torch.nn as nn
from typing import Dict, Any


class LearnableWaveletDWT1d(nn.Module):
    """
    Discrete Wavelet Transform (DWT) 1D Filterbank.
    Decomposes multi-channel IMU time-series into:
    - Low-Frequency Approximation (Trend / Macroscopic Kinematics)
    - High-Frequency Detail (Vibration / Mechanical Shock / Engine Rumble)
    
    Mathematically formulated as Depthwise Conv1D with stride=2,
    making it 100% natively compatible with TensorFlow Lite and mobile NPUs.
    """
    def __init__(self, in_channels: int = 12):
        super().__init__()
        self.in_channels = in_channels
        
        # Initialize with standard Haar / Daubechies orthogonal filters
        # Low-pass filter h0: [1/sqrt(2), 1/sqrt(2)]
        # High-pass filter h1: [1/sqrt(2), -1/sqrt(2)]
        h0 = torch.tensor([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)], dtype=torch.float32)
        h1 = torch.tensor([1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)], dtype=torch.float32)
        
        filters = torch.stack([h0, h1])  # (2, 2)
        # Replicate across in_channels: (in_channels * 2, 1, 2)
        weight = torch.zeros(in_channels * 2, 1, 2)
        for i in range(in_channels):
            weight[i * 2, 0] = h0
            weight[i * 2 + 1, 0] = h1
            
        self.dwt_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels * 2,
            kernel_size=2,
            stride=2,
            padding=0,
            groups=in_channels,
            bias=False
        )
        self.dwt_conv.weight = nn.Parameter(weight, requires_grad=True)
        self.bn = nn.BatchNorm1d(in_channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 12, 450)
        out = self.dwt_conv(x)  # (B, 24, 225)
        return self.bn(out)


class SqueezeAndExcitation1d(nn.Module):
    """
    Channel Attention Module: Dynamically recalibrates the importance 
    of approximation vs. detail sub-bands and sensor channels.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        weights = self.fc(x).view(b, c, 1)
        return x * weights


class WaveletMultiScaleBlock(nn.Module):
    """
    Multi-Scale Feature Extraction Block processing wavelet-transformed sub-bands
    across parallel temporal receptive fields (k=3, 7, 15).
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        branch_channels = out_channels // 3
        remainder = out_channels - (branch_channels * 2)

        # Branch 1: High frequency vibration resolution (k=3)
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # Branch 2: Medium frequency bounce resolution (k=7)
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=7, stride=stride, padding=3, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # Branch 3: Low frequency macro kinematics (k=15)
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, remainder, kernel_size=15, stride=stride, padding=7, bias=False),
            nn.BatchNorm1d(remainder),
            nn.ReLU(inplace=True)
        )

        self.se = SqueezeAndExcitation1d(out_channels)

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        out = torch.cat([b1, b2, b3], dim=1)
        out = self.se(out)
        return self.act(out + res)


class StageSkipProjection(nn.Module):
    """
    Skip projection module inspired by Delli Priscoli et al. (Sensors 2020 / PMC7767000):
    Projects multi-scale convolutional feature maps directly to a fixed-dimensional
    representation via dual pooling (GAP + GMP) before multi-level concatenation.
    """
    def __init__(self, in_channels: int, out_dim: int = 40, dropout: float = 0.2):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_p = self.gap(x)
        max_p = torch.amax(x, dim=2, keepdim=True)
        pooled = torch.cat([avg_p, max_p], dim=1)
        return self.proj(pooled)


class MoveMintWaveletSCNN(nn.Module):
    """
    Hierarchical Wavelet Multi-Scale S-CNN (H-WSCNN)
    Combines:
    1. Discrete Wavelet Transform (DWT) front-end (Haar low-pass trend + high-pass detail)
    2. Multi-Scale parallel receptive fields (k=3, 7, 15)
    3. Squeeze-and-Excitation channel attention
    4. Hierarchical Skip Forwarding (PMC7767000 / Delli Priscoli et al.):
       Taps low-level (Stage 1), mid-level (Stage 2), and deep (Stage 3) feature representations
       and fuses them via parallel 40-dim skip projections into a 120-dim unified latent vector.
    5. Dual pooling (GAP + GMP) for steady vibration and sudden jerk preservation.
    
    Input Shape:  (Batch, in_channels, L)  [Default: (Batch, 12, 450)]
    Output Shape: (Batch, num_classes)     [Default: (Batch, 8)]
    """
    def __init__(self, in_channels: int = 12, num_classes: int = 8, dropout_rate: float = 0.3):
        super().__init__()

        # Stage 0: Wavelet Decomposition Stem
        self.wavelet_stem = LearnableWaveletDWT1d(in_channels=in_channels)  # (B, C, L) -> (B, C*2, L/2)
        
        stem_out_dim = in_channels * 2
        self.stem_proj = nn.Sequential(
            nn.Conv1d(stem_out_dim, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        # Stage 1: Shallow micro-vibrations (64 channels)
        self.stage1 = nn.Sequential(
            WaveletMultiScaleBlock(64, 64, stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.skip_stage1 = StageSkipProjection(64, out_dim=40, dropout=dropout_rate * 0.7)

        # Stage 2: Mid-level sway & bump dynamics (128 channels)
        self.stage2 = nn.Sequential(
            WaveletMultiScaleBlock(64, 128, stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.skip_stage2 = StageSkipProjection(128, out_dim=40, dropout=dropout_rate * 0.7)

        # Stage 3: Deep macro-transit kinematics (128 channels)
        self.stage3 = nn.Sequential(
            WaveletMultiScaleBlock(128, 128, stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.skip_stage3 = StageSkipProjection(128, out_dim=40, dropout=dropout_rate * 0.7)

        # Hierarchical Classifier Head: Fuses 3 x 40 = 120 dims
        self.head = nn.Sequential(
            nn.Linear(120, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        s0 = self.stem_proj(self.wavelet_stem(x))
        s1 = self.stage1(s0)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)

        # Hierarchical skip projections (PMC7767000)
        p1 = self.skip_stage1(s1)  # (B, 40)
        p2 = self.skip_stage2(s2)  # (B, 40)
        p3 = self.skip_stage3(s3)  # (B, 40)

        # Fused multi-level latent representation: (B, 120)
        fused = torch.cat([p1, p2, p3], dim=1)
        logits = self.head(fused)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_summary(self) -> Dict[str, Any]:
        total_params = self.count_parameters()
        estimated_size_kb = (total_params * 4) / 1024
        return {
            "total_trainable_parameters": total_params,
            "estimated_fp32_size_kb": round(estimated_size_kb, 2),
            "estimated_int8_size_kb": round(estimated_size_kb / 4, 2)
        }


# Model alias for semantic clarity
HierarchicalWaveletSCNN = MoveMintWaveletSCNN


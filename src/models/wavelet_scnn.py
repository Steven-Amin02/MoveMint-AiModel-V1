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


class MoveMintWaveletSCNN(nn.Module):
    """
    Wavelet-Integrated Multi-Scale S-CNN (W-SCNN)
    Combines:
    1. Discrete Wavelet Transform (DWT) front-end (Low-Pass Trend + High-Pass Detail)
    2. Multi-Scale parallel receptive fields
    3. Squeeze-and-Excitation channel attention
    4. Dual pooling (GAP + GMP) to capture both steady energy and peak jerks
    
    Input Shape:  (Batch, 12, 450)
    Output Shape: (Batch, 8)
    """
    def __init__(self, in_channels: int = 12, num_classes: int = 8, dropout_rate: float = 0.3):
        super().__init__()

        # Stage 0: Wavelet Decomposition Stem
        self.wavelet_stem = LearnableWaveletDWT1d(in_channels=in_channels)  # (B, 12, 450) -> (B, 24, 225)
        
        self.stem_proj = nn.Sequential(
            nn.Conv1d(24, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        # Stage 1: Feature learning at 64 channels
        self.stage1 = nn.Sequential(
            WaveletMultiScaleBlock(64, 64, stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2)  # (64, 112)
        )

        # Stage 2: Feature learning at 128 channels
        self.stage2 = nn.Sequential(
            WaveletMultiScaleBlock(64, 128, stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2)  # (128, 56)
        )

        # Stage 3: Deep refinement at 128 channels
        self.stage3 = nn.Sequential(
            WaveletMultiScaleBlock(128, 128, stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2)  # (128, 28)
        )

        # Global Pooling: Global Average + Global Max (torch.amax)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Dense Classifier Head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(128 * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 12, 450)
        x = self.wavelet_stem(x)  # (B, 24, 225)
        x = self.stem_proj(x)     # (B, 64, 225)
        x = self.stage1(x)        # (B, 64, 112)
        x = self.stage2(x)        # (B, 128, 56)
        x = self.stage3(x)        # (B, 128, 28)

        # Dual pooling: Average energy + Peak max
        avg_pool = self.gap(x)                           # (B, 128, 1)
        max_pool = torch.amax(x, dim=2, keepdim=True)    # (B, 128, 1) - clean ONNX ReduceMax
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # (B, 256, 1)

        logits = self.head(pooled)
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

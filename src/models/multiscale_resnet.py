import torch
import torch.nn as nn
from typing import Dict, Any


class SqueezeAndExcitation1d(nn.Module):
    """
    Channel Attention (SE) Block for 1D sensor time-series.
    Dynamically recalibrates the importance of the 12 sensor channels
    (e.g., boosting Gyroscope during turns, Accelerometer during road vibrations).
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced_channels = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        b, c, _ = x.size()
        weights = self.fc(x).view(b, c, 1)  # (B, C, 1)
        return x * weights


class MultiScaleResBlock1d(nn.Module):
    """
    Multi-Scale Residual Block with parallel receptive fields:
    - Small kernel (k=3): High-frequency chassis & road vibration
    - Medium kernel (k=7): Kinematic suspension bounce & motor rumble
    - Large kernel (k=15): Macro stop-and-go acceleration & braking profiles
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        branch_channels = out_channels // 3
        remainder = out_channels - (branch_channels * 2)

        # Branch 1: High frequency (k=3)
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # Branch 2: Medium frequency (k=7)
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=7, stride=stride, padding=3, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # Branch 3: Macro kinematics (k=15)
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, remainder, kernel_size=15, stride=stride, padding=7, bias=False),
            nn.BatchNorm1d(remainder),
            nn.ReLU(inplace=True)
        )

        # Squeeze-and-Excitation Channel Attention
        self.se = SqueezeAndExcitation1d(out_channels)

        # Residual shortcut connection
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


class MoveMintMultiScaleResNet(nn.Module):
    """
    Advanced Multi-Scale ResNet with Squeeze-and-Excitation attention
    designed specifically for vehicle and transit vibration discrimination.
    Input Shape: (Batch, 12, 450)
    Output Shape: (Batch, 8)
    """
    def __init__(self, in_channels: int = 12, num_classes: int = 8, dropout_rate: float = 0.3):
        super().__init__()

        # Stem: Initial projection from 12 sensor channels
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)  # (32, 225)
        )

        # Stage 1: Feature learning at 64 channels
        self.stage1 = nn.Sequential(
            MultiScaleResBlock1d(32, 64, stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2)  # (64, 112)
        )

        # Stage 2: Feature learning at 128 channels
        self.stage2 = nn.Sequential(
            MultiScaleResBlock1d(64, 128, stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2)  # (128, 56)
        )

        # Stage 3: Feature learning at 128 channels
        self.stage3 = nn.Sequential(
            MultiScaleResBlock1d(128, 128, stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2)  # (128, 28)
        )

        # Global Average Pooling (Temporal invariance)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Classification Head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        logits = self.head(x)
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

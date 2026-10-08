import torch
import torch.nn as nn
from typing import Dict, Any


class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise Separable 1D Convolution Block.
    Dramatically reduces parameter count and FLOPs for mobile ARM deployment.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 1, padding: int = 2):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False
        )
        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class MoveMintEdgeCNN(nn.Module):
    """
    Lightweight 1D-CNN designed for low-power edge inference on Android via TFLite.
    Input Shape: (Batch, 12, 450)
    Output Shape: (Batch, 8)
    """
    def __init__(self, in_channels: int = 12, num_classes: int = 8, dropout_rate: float = 0.3):
        super().__init__()
        
        # Stem Block: Captures initial cross-channel sensor correlations
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)  # (32, 225)
        )

        # Backbone: Depthwise Separable Blocks
        self.block1 = nn.Sequential(
            DepthwiseSeparableConv1d(32, 64, kernel_size=5, padding=2),
            nn.MaxPool1d(kernel_size=2, stride=2)  # (64, 112)
        )

        self.block2 = nn.Sequential(
            DepthwiseSeparableConv1d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool1d(kernel_size=2, stride=2)  # (128, 56)
        )

        self.block3 = nn.Sequential(
            DepthwiseSeparableConv1d(128, 128, kernel_size=3, padding=1),
            nn.MaxPool1d(kernel_size=2, stride=2)  # (128, 28)
        )

        # Global Average Pooling: Translation invariance & ultra-compact head
        self.gap = nn.AdaptiveAvgPool1d(1)  # (128, 1)

        # Classifier Head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 12, 450)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        logits = self.head(x)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_summary(self) -> Dict[str, Any]:
        total_params = self.count_parameters()
        estimated_size_kb = (total_params * 4) / 1024  # 4 bytes per float32
        return {
            "total_trainable_parameters": total_params,
            "estimated_fp32_size_kb": round(estimated_size_kb, 2),
            "estimated_int8_size_kb": round(estimated_size_kb / 4, 2)
        }

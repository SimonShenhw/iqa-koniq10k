"""
基线 CNN / Baseline CNN.

设计目的 / Purpose:
    故意保持简单 —— 4 个 (Conv-BN-ReLU-MaxPool) 块 + GAP + FC。
    作为对比基线，让 ResNet-50/ViT 的提升有可量化的参照。

    Intentionally minimal — 4 (Conv→BN→ReLU→MaxPool) blocks, then GAP + FC.
    Serves as the comparison foil so gains from ResNet-50 / ViT are measurable.

架构 / Architecture:
    Input [B, 3, 224, 224]
      → Conv(3→32)  → BN → ReLU → MaxPool 2x   [B, 32, 112, 112]
      → Conv(32→64) → BN → ReLU → MaxPool 2x   [B, 64, 56, 56]
      → Conv(64→128)→ BN → ReLU → MaxPool 2x   [B,128, 28, 28]
      → Conv(128→256)→ BN→ ReLU → MaxPool 2x   [B,256, 14, 14]
      → GlobalAvgPool                          [B, 256]
      → FC(256 → num_buckets 或 1)             [B, num_buckets] / [B, 1]

参见技术文档 Section 6.1 / See spec Section 6.1.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """单个 Conv-BN-ReLU-MaxPool 块 / single Conv-BN-ReLU-MaxPool block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))


class BaselineCNN(nn.Module):
    """基线 CNN，可切换 MSE 标量输出或分布输出。

    Baseline CNN supporting both MSE scalar output and distribution output.

    Args:
        num_buckets:       分布模式下的桶数 / bins in distribution mode
        distribution_mode: True→softmax 分布，False→单标量 MOS 回归
                           True → softmax distribution, False → scalar MSE regression
    """

    def __init__(
        self,
        num_buckets: int = 10,
        distribution_mode: bool = False,
        **_unused,  # 兼容工厂传入多余参数 / accept extra kwargs from factory
    ):
        super().__init__()
        self.distribution_mode = distribution_mode

        # 特征提取 / feature extractor — 32 → 64 → 128 → 256 通道
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )

        # 全局平均池化 —— 把 [B, 256, H, W] 压成 [B, 256]
        # Global average pooling — [B, 256, H, W] → [B, 256]
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 分类 / 回归头 / head
        out_dim = num_buckets if distribution_mode else 1
        self.head = nn.Linear(256, out_dim)

        # 分布模式用 softmax 输出概率 / softmax in distribution mode
        self.softmax = nn.Softmax(dim=-1) if distribution_mode else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)                 # [B, 256, H/16, W/16]
        x = self.gap(x).flatten(1)           # [B, 256]
        x = self.head(x)                     # [B, num_buckets] or [B, 1]
        return self.softmax(x)

    # ------------------------------------------------------------------
    # Grad-CAM hook 目标层（若扩展成 CAM 可视化时使用）
    # Target layer for Grad-CAM (if extended for CAM visualization)
    # ------------------------------------------------------------------
    def gradcam_target_layer(self) -> nn.Module:
        """最后一个卷积块，用于 Grad-CAM / last conv block for Grad-CAM."""
        return self.features[-1].conv

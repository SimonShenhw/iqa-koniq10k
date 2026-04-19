"""
ResNet-50 + EMD 主力模型 / ResNet-50 + EMD primary model.

设计 / Design:
    - torchvision ResNet-50, ImageNet-1K V2 预训练权重
    - fc 层替换为 Linear(2048, num_buckets) + Softmax
    - 训练采用两阶段冻结策略：先只训 head，再解冻 layer3/layer4

    - torchvision ResNet-50 with IMAGENET1K_V2 pretrained weights
    - Replace fc with Linear(2048, num_buckets) + Softmax
    - Two-phase freezing: head-only first, then unfreeze layer3 & layer4

训练策略 / Training recipe:
    Phase 1 (epochs 0-4):  只训 head，LR=1e-3
                            head-only, LR=1e-3
    Phase 2 (epochs 5-29): 解冻 layer3/layer4，LR 降到 1e-5 主干 / 1e-4 head
                            unfreeze layer3/layer4, backbone LR=1e-5, head LR=1e-4

Grad-CAM 目标层 / Grad-CAM target layer: model.layer4[-1]

参见技术文档 Section 6.2 / See spec Section 6.2.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torchvision.models as tv_models


class ResNet50EMD(nn.Module):
    """ResNet-50 + 分布预测 head / ResNet-50 + distribution head.

    Args:
        num_buckets:       输出桶数 / number of output bins
        pretrained:        是否加载 ImageNet 预训练权重 / load IMAGENET1K_V2 weights
        distribution_mode: True 时输出 softmax 分布 / softmax output if True
    """

    def __init__(
        self,
        num_buckets: int = 10,
        pretrained: bool = True,
        distribution_mode: bool = True,
        **_unused,  # 兼容工厂 / swallow extra kwargs
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.distribution_mode = distribution_mode

        # 加载主干 / Load backbone
        # V2 权重的 ImageNet Top-1 约 80.9%，优于 V1 的 76.1%
        # V2 weights give ~80.9% ImageNet top-1 vs 76.1% for V1
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = tv_models.resnet50(weights=weights)

        # 替换 fc：2048 → num_buckets / replace fc head
        # 训练时对应的 softmax 放在 forward 里，以便 XAI / compile 能看到
        self.backbone.fc = nn.Linear(2048, num_buckets if distribution_mode else 1)

        # Softmax only in distribution mode / 仅分布模式下走 softmax
        self.softmax = nn.Softmax(dim=-1) if distribution_mode else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)     # [B, num_buckets] or [B, 1]
        return self.softmax(logits)

    # ==================================================================
    # 冻结 / 解冻工具 / freeze & unfreeze helpers
    # ==================================================================
    def freeze_backbone_except_fc(self) -> None:
        """Phase 1: 冻结除 fc 以外的所有参数 / freeze everything except fc."""
        for name, param in self.backbone.named_parameters():
            param.requires_grad = name.startswith("fc.")

    def unfreeze_layers(self, layer_names: List[str]) -> None:
        """Phase 2: 解冻指定层 / unfreeze named layers.

        典型用法 / typical use:
            model.unfreeze_layers(["layer3", "layer4"])
        """
        for name, param in self.backbone.named_parameters():
            # 已解冻的 fc 保持原状 / leave fc alone (already trainable)
            if any(name.startswith(layer + ".") for layer in layer_names):
                param.requires_grad = True

    def trainable_parameter_groups(
        self, lr_head: float, lr_backbone: float
    ) -> List[dict]:
        """构造双学习率参数组，供 optimizer 使用 / build two-LR param groups.

        Returns:
            [
              {"params": [head params],     "lr": lr_head},
              {"params": [backbone params], "lr": lr_backbone},
            ]
            仅包含 requires_grad=True 的参数 / only params with requires_grad=True
        """
        head_params, backbone_params = [], []
        for name, param in self.backbone.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("fc."):
                head_params.append(param)
            else:
                backbone_params.append(param)

        groups = []
        if head_params:
            groups.append({"params": head_params, "lr": lr_head})
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr_backbone})
        return groups

    # ------------------------------------------------------------------
    # XAI 钩子 / XAI hooks
    # ------------------------------------------------------------------
    def gradcam_target_layer(self) -> nn.Module:
        """Grad-CAM 最佳目标层：layer4 最后一个 bottleneck.

        Best Grad-CAM target layer: the last bottleneck of layer4.
        空间分辨率为 7×7，语义最丰富 / 7×7 feature map, most semantic.
        """
        return self.backbone.layer4[-1]

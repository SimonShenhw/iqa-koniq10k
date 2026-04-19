"""模型模块 / Models module.

提供 / Exposes:
- BaselineCNN: 基线 CNN (MSE) / baseline CNN
- ResNet50EMD: 主力模型 / primary distribution-prediction model
- ViTLoRA: 参数高效微调（需 timm+peft）/ PEFT model (requires timm+peft)
- build_model: 工厂函数 / factory dispatcher

注 / Note:
    ViTLoRA 使用延迟 import —— 只有在真正构造时才要求 timm/peft。
    Lazy import for ViTLoRA — timm/peft only required when actually built.
"""

from __future__ import annotations

from typing import Any

from .baseline_cnn import BaselineCNN
from .resnet50_emd import ResNet50EMD


def build_model(name: str, **kwargs: Any):
    """根据配置名称构建模型 / Build a model by name.

    Args:
        name: 'baseline_cnn' | 'resnet50_emd' | 'vit_lora'
        **kwargs: 传入具体模型 / forwarded to the model constructor
    """
    name = name.lower()
    if name == "baseline_cnn":
        return BaselineCNN(**kwargs)
    if name == "resnet50_emd":
        return ResNet50EMD(**kwargs)
    if name == "vit_lora":
        # 延迟 import —— 缺 timm/peft 也不影响其他模型
        # Lazy import — missing timm/peft doesn't break other models
        from .vit_lora import ViTLoRA
        return ViTLoRA(**kwargs)
    raise ValueError(f"未知模型 / Unknown model: {name}")


def __getattr__(name: str):
    """延迟属性访问：import ViTLoRA 时才触发 / lazy attribute access."""
    if name == "ViTLoRA":
        from .vit_lora import ViTLoRA
        return ViTLoRA
    raise AttributeError(f"module 'src.models' has no attribute {name!r}")


__all__ = ["BaselineCNN", "ResNet50EMD", "ViTLoRA", "build_model"]

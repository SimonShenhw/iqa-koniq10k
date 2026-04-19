"""XAI 可视化模块 / Explainable AI module.

- Grad-CAM: 针对 CNN，需 pytorch-grad-cam / for CNN, requires pytorch-grad-cam
- Attention Rollout: 针对 ViT，纯 torch 实现 / pure-torch

两者均延迟 import，缺失依赖不会阻塞其他模块。
Both are lazy-imported — missing deps won't break other modules.
"""

from __future__ import annotations


def __getattr__(name: str):
    """延迟 import，缺依赖时只有访问时才报错 / lazy import on attribute access."""
    if name == "ResNetGradCAM":
        from .gradcam import ResNetGradCAM
        return ResNetGradCAM
    if name == "attention_rollout":
        from .attention_rollout import attention_rollout
        return attention_rollout
    if name == "register_attention_hooks":
        from .attention_rollout import register_attention_hooks
        return register_attention_hooks
    raise AttributeError(f"module 'src.xai' has no attribute {name!r}")


__all__ = ["ResNetGradCAM", "attention_rollout", "register_attention_hooks"]

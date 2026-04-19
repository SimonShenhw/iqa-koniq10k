"""
ViT-B/16 + LoRA + EMD 模型 / ViT-B/16 + LoRA + EMD model.

核心要点 / Key points:
    - timm 的 vit_base_patch16_224 主干 + ImageNet-21k 预训练
    - LoRA 仅作用于 qkv（r=16, alpha=32），<1% 参数可训练
    - EMD 分布输出（与 ResNet-50 版本共用 head 结构）

    - timm vit_base_patch16_224 backbone + ImageNet-21k pretrain
    - LoRA on qkv only (r=16, alpha=32), <1% trainable params
    - EMD distribution output (same head pattern as ResNet-50 version)

⚠️ LoRA target_modules 注意 / LoRA target_modules caveat:
    timm ViT-B/16 把 Q/K/V 融合为单个 Linear，模块名为 'qkv'。
    换模型前务必先打印 named_modules 确认；如 eva02 / DeiT 等可能拆成 q、k、v。

    timm's ViT-B/16 fuses Q/K/V into a single Linear named 'qkv'. Before
    switching backbone (eva02, DeiT, etc.) always print named_modules to
    verify — they may split into separate q/k/v Linears.

参见技术文档 Section 6.3 / See spec Section 6.3.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

# timm / peft 为可选依赖：延迟 import，错误信息更清晰
# timm / peft are optional — defer import for clearer error message
try:
    import timm
except ImportError as e:
    raise ImportError(
        "timm 未安装 / timm is required. Run: pip install timm>=1.0"
    ) from e

try:
    from peft import LoraConfig, get_peft_model
except ImportError as e:
    raise ImportError(
        "peft 未安装 / peft is required. Run: pip install peft>=0.11"
    ) from e


class ViTLoRA(nn.Module):
    """ViT-B/16 + LoRA + 分布预测 head.

    Args:
        num_buckets:         输出桶数 / output bins
        pretrained:          是否加载预训练权重 / load pretrained weights
        backbone:            timm 模型名 / timm model name
        lora_r:              LoRA 秩 / LoRA rank
        lora_alpha:          LoRA 缩放 / LoRA scaling
        lora_dropout:        LoRA dropout
        lora_target_modules: 要注入 LoRA 的模块名 / target module names
        lora_modules_to_save: 需全量训练的模块 / modules trained fully (e.g. head)
        distribution_mode:   True 时 softmax 输出 / softmax if True
    """

    def __init__(
        self,
        num_buckets: int = 10,
        pretrained: bool = True,
        backbone: str = "vit_base_patch16_224",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: List[str] = None,
        lora_modules_to_save: List[str] = None,
        distribution_mode: bool = True,
        **_unused,
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.distribution_mode = distribution_mode

        # ---- 1. 构建 timm ViT 主干（带 classifier）----
        # ---- 1. Build timm ViT backbone with classifier head ----
        # num_classes = num_buckets → timm 自动构造 Linear(768, num_buckets)
        # num_classes=num_buckets → timm builds Linear(768, num_buckets) for us
        out_dim = num_buckets if distribution_mode else 1
        base_model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=out_dim,
        )

        # ---- 2. 配置 LoRA / configure LoRA ----
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules or ["qkv"],
            lora_dropout=lora_dropout,
            bias="none",
            # 新 head 需全量训练，不走 LoRA / train head fully, bypass LoRA
            modules_to_save=lora_modules_to_save or ["head"],
        )

        # ---- 3. 包装成 PEFT 模型 / wrap as PEFT model ----
        self.peft_model = get_peft_model(base_model, lora_cfg)

        self.softmax = nn.Softmax(dim=-1) if distribution_mode else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.peft_model(x)   # [B, num_buckets] or [B, 1]
        return self.softmax(logits)

    # ==================================================================
    # 实用工具 / utilities
    # ==================================================================
    def print_trainable_parameters(self) -> None:
        """打印可训练参数比例 / print trainable param percentage.

        预期 / Expected for ViT-B/16 + LoRA(r=16):
            trainable params: ~850K || all params: ~86M || trainable%: ~0.99
        """
        self.peft_model.print_trainable_parameters()

    @staticmethod
    def audit_attention_modules(backbone_name: str = "vit_base_patch16_224") -> None:
        """调试工具：打印含 'qkv' 或 'attn' 的模块名 / debug: list attn modules.

        在换主干或验证 LoRA target_modules 时调用。
        Call this when switching backbones or verifying LoRA target_modules.
        """
        model = timm.create_model(backbone_name, pretrained=False)
        print(f"=== {backbone_name} 注意力模块审计 / attention module audit ===")
        for name, module in model.named_modules():
            if "qkv" in name or "attn" in name:
                print(f"  {name}: {type(module).__name__}")

    def collect_attentions_for_rollout(self, x: torch.Tensor) -> List[torch.Tensor]:
        """前向一次并收集每层注意力矩阵，用于 Attention Rollout.

        Run a forward pass and collect per-layer attention matrices for
        Attention Rollout visualization.

        注意 / Note:
            timm ViT 默认不导出注意力。这里用 hook 捕获每个 block 的 attn.softmax 输出。
            timm ViT doesn't expose attentions by default — we register hooks
            on each block's attention softmax output.
        """
        attentions: List[torch.Tensor] = []
        hooks = []

        def _hook(_module, _inp, output):
            # output 通常为 (attn_output, attn_weights) 或仅 attn_output
            # timm 不同版本导出形式不同，这里做健壮处理
            if isinstance(output, tuple) and len(output) >= 2:
                attentions.append(output[1].detach())

        # 在 PEFT 包装外找到原始 ViT blocks
        # Find underlying ViT blocks (PEFT wraps the model in .base_model)
        base = getattr(self.peft_model, "base_model", self.peft_model)
        inner = getattr(base, "model", base)
        for blk in inner.blocks:
            hooks.append(blk.attn.register_forward_hook(_hook))

        try:
            _ = self(x)
        finally:
            for h in hooks:
                h.remove()
        return attentions

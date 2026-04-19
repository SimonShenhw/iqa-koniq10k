"""
Attention Rollout 可视化（针对 ViT）/ Attention Rollout for ViT.

参考 / Reference:
    Abnar & Zuidema. "Quantifying Attention Flow in Transformers." ACL 2020.

算法 / Algorithm:
    1) 每层注意力：A_l = softmax(QK^T / sqrt(d))，shape [B, heads, N+1, N+1]
    2) 多头取平均：Ā_l = mean over heads → [B, N+1, N+1]
    3) 残差近似：Ã_l = Ā_l + I，再行归一化
       residual approximation: Ã_l = Ā_l + I, then row-normalize
    4) 逐层连乘：R = Ã_L · Ã_{L-1} · ... · Ã_1
    5) 取 CLS → patches 的行：R[:, 0, 1:]，reshape 成 14×14 热力图

为什么能可视化 IQA 关注点？/ Why this visualizes IQA attention?
    CLS token 是分类头的直接输入，R[:, 0, 1:] 表示"CLS 最终聚合了哪些 patch 的信息"，
    相当于模型对图像各区域的注意力权重。
    CLS is the classification head's input, so R[:, 0, 1:] reveals which patches
    the CLS token ultimately pulled information from.

参见技术文档 Section 9.2 / See spec Section 9.2.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _find_vit_blocks(model: nn.Module):
    """搜索底层 timm ViT 的 blocks 容器 / locate timm ViT blocks.

    用 named_modules 遍历整棵树，找到首个名字以 'blocks' 结尾、是 Sequential
    或 ModuleList、且首个子 block 含 .attn 属性的节点。

    Walk named_modules; return the first node whose name ends in 'blocks',
    is either Sequential or ModuleList, and whose first child has .attn.

    为什么不用 isinstance(..., ModuleList)？/ Why not ModuleList check?
        timm 1.0+ 把 blocks 定义为 nn.Sequential（便于 torch.compile 和
        batch_size 变化）。严格限定 ModuleList 会误判。
        timm 1.0+ uses nn.Sequential for blocks (better for compile + shape
        changes). A strict ModuleList check would miss it.
    """
    for name, mod in model.named_modules():
        # 只考虑"叫 blocks"的容器节点 / only consider containers named 'blocks'
        if not (name == "blocks" or name.endswith(".blocks")):
            continue
        if not isinstance(mod, (nn.Sequential, nn.ModuleList)):
            continue
        children = list(mod.children())
        if not children:
            continue
        # 首个 block 需有 .attn 子模块（才能 hook）
        # first child must have an .attn submodule so we can hook it
        if hasattr(children[0], "attn"):
            return mod

    raise AttributeError(
        "未找到 ViT blocks / could not locate ViT .blocks — "
        "model 结构不兼容 / incompatible model structure"
    )


class _FusedAttnRestorer:
    """伪 hook handle：在 .remove() 时恢复 attn.fused_attn 原值.

    Pseudo hook handle: restores attn.fused_attn on .remove(). Allows callers
    to keep the uniform `for h in hooks: h.remove()` pattern.
    """

    def __init__(self, attn_module, original_value: bool):
        self._mod = attn_module
        self._original = original_value

    def remove(self) -> None:
        self._mod.fused_attn = self._original


def register_attention_hooks(model: nn.Module):
    """在每个 ViT block 的 attn_drop 模块上注册 pre-hook，捕获 softmax 后的注意力.

    Register pre-forward hooks on each block's `attn.attn_drop` to capture
    softmax-normalized attention weights.

    ⚠️ 关键细节 / Critical detail:
        timm 1.0+ 的 Attention 模块默认 fused_attn=True，会调用 PyTorch 内置
        的 F.scaled_dot_product_attention 融合核，整个注意力计算走 C++ 内核，
        attn_drop 模块不会被调用、pre-hook 永远不触发。

        本函数在注册 hook 前把每个 block.attn.fused_attn 暂时置为 False，
        以便走回 "q@k.T → softmax → attn_drop(attn) → @v" 的 Python 路径。
        返回的 hooks 列表里包含 _FusedAttnRestorer，用户调用 h.remove() 时
        自动恢复原值，保持推理速度不受影响。

        timm 1.0+ Attention defaults to fused_attn=True, which routes
        everything through F.scaled_dot_product_attention — a C++ fused
        kernel that skips the attn_drop module entirely, so pre-hooks never
        fire. We temporarily set fused_attn=False on each block before
        registering hooks, and include _FusedAttnRestorer objects in the
        returned list so `for h in hooks: h.remove()` restores the original
        values. Inference speed outside visualization is unaffected.

    Args:
        model: 原始 timm ViT / PEFT 包装 / 我们的 ViTLoRA 均可
               raw timm ViT, PEFT-wrapped model, or our ViTLoRA — all OK

    Returns:
        (attn_storage, hooks):
            attn_storage: list[Tensor]，forward 后每层注意力逐个 append
            hooks:        句柄列表，用完调用 handle.remove()
    """
    attn_storage: List[torch.Tensor] = []
    hooks = []

    def _make_pre_hook():
        # pre-hook 签名 (module, args) / pre-hook signature (module, args)
        def _hook(_module, args):
            # Dropout 的 args 是 (attn_weights,) / args for Dropout = (attn_weights,)
            if args:
                attn_storage.append(args[0].detach())
        return _hook

    blocks = _find_vit_blocks(model)

    for blk in blocks:
        attn_mod = blk.attn
        # 暂关 fused_attn 并记录原值以便恢复
        # Temporarily disable fused_attn and remember original for restoration
        original_fused = getattr(attn_mod, "fused_attn", False)
        if original_fused:
            attn_mod.fused_attn = False
            hooks.append(_FusedAttnRestorer(attn_mod, original_fused))

        # pre-hook on attn_drop
        hooks.append(attn_mod.attn_drop.register_forward_pre_hook(_make_pre_hook()))

    return attn_storage, hooks


def attention_rollout(attentions: List[torch.Tensor]) -> torch.Tensor:
    """从每层注意力计算 CLS → patch 的 rollout 热力图.

    Args:
        attentions: 每层 shape [B, heads, N+1, N+1] 的注意力列表
                    list of per-layer attention [B, heads, N+1, N+1]

    Returns:
        [B, 14, 14] rollout 热力图（未上采样）/ rollout heatmap (before upsample)
    """
    if not attentions:
        raise ValueError("attentions 为空 / empty attention list")

    device = attentions[0].device
    n_tokens = attentions[0].size(-1)  # N+1（含 CLS）/ N+1 incl. CLS

    # result 起点是单位阵，尺寸 [N+1, N+1] / start with identity
    result = torch.eye(n_tokens, device=device).unsqueeze(0)  # [1, N+1, N+1]
    result = result.expand(attentions[0].size(0), -1, -1)     # [B, N+1, N+1]

    eye = torch.eye(n_tokens, device=device).unsqueeze(0)     # 残差矩阵 I / residual I

    for attn in attentions:
        # 头维度取平均 / average over heads
        attn_mean = attn.mean(dim=1)  # [B, N+1, N+1]

        # 残差近似 + 行归一化 / residual + row-normalize
        attn_residual = attn_mean + eye
        attn_normalized = attn_residual / attn_residual.sum(dim=-1, keepdim=True)

        # 累乘 / cumulative matmul
        result = torch.bmm(attn_normalized, result)

    # CLS → patches，跳过 CLS-to-CLS / skip CLS-to-CLS
    cls_attention = result[:, 0, 1:]  # [B, N]

    # 重排为 14×14 / reshape to 14×14
    # ViT-B/16 + 224 输入 → 14×14 patches
    grid_size = int(cls_attention.size(-1) ** 0.5)
    if grid_size * grid_size != cls_attention.size(-1):
        raise ValueError(
            f"patch 数 {cls_attention.size(-1)} 非完全平方数 / non-square patch count"
        )
    return cls_attention.reshape(-1, grid_size, grid_size)


def rollout_heatmap_to_image_size(
    rollout: torch.Tensor,
    image_size: int = 224,
) -> torch.Tensor:
    """双线性上采样 rollout 到输入图像大小 / bilinear upsample rollout to image size.

    Args:
        rollout:    [B, 14, 14] 原始 rollout / raw rollout
        image_size: 目标边长 / target spatial size

    Returns:
        [B, image_size, image_size] 归一到 [0,1] 的热力图 / [0,1]-normalized heatmap
    """
    # 增加通道维 → 上采样 → 去通道维
    # Add channel dim → upsample → squeeze
    up = F.interpolate(
        rollout.unsqueeze(1),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    # 按样本独立归一到 [0, 1] / min-max normalize per-sample
    B = up.size(0)
    flat = up.reshape(B, -1)
    min_v = flat.min(dim=1, keepdim=True).values
    max_v = flat.max(dim=1, keepdim=True).values
    return ((flat - min_v) / (max_v - min_v + 1e-8)).reshape(B, image_size, image_size)

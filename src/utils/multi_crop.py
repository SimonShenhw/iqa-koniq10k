"""
测试时增强 / Test-Time Augmentation (TTA).

技术文档要求 5-crop 或 25-crop 平均概率分布后再算期望。
在验证/测试阶段对每张图生成多个 crop，模型输出分布取平均再算 SRCC。

Spec requires 5-crop or 25-crop averaging — for each test image we generate
multiple crops, run inference, average the output distributions, then compute
the expected score. This typically adds ~1-2 SRCC points.

参见技术文档 Section 3 与 Section 8 / See spec Sections 3 and 8.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def five_crop(image: torch.Tensor, crop_size: int = 224) -> torch.Tensor:
    """5-crop：4 角 + 中心 / 5-crop: 4 corners + center.

    Args:
        image:     [C, H, W] 或 [B, C, H, W] 单张/批次图像 / single or batched image
        crop_size: 正方形 crop 边长 / square crop size

    Returns:
        [5, C, crop_size, crop_size] 或 [B, 5, C, crop_size, crop_size]
    """
    if image.dim() == 3:
        c, h, w = image.shape
        batched = False
    elif image.dim() == 4:
        # 批次处理：逐张切，再 stack / batched — crop per image then stack
        return torch.stack([five_crop(img, crop_size) for img in image], dim=0)
    else:
        raise ValueError(f"期望 3 或 4 维 / expected 3- or 4-D, got {image.dim()}-D")

    assert h >= crop_size and w >= crop_size, (
        f"图像尺寸 {h}x{w} 小于 crop {crop_size} / image smaller than crop"
    )

    # 5 个左上角坐标 / 5 top-left coordinates
    h_mid = (h - crop_size) // 2
    w_mid = (w - crop_size) // 2
    coords = [
        (0, 0),                      # 左上 / top-left
        (0, w - crop_size),          # 右上 / top-right
        (h - crop_size, 0),          # 左下 / bottom-left
        (h - crop_size, w - crop_size),  # 右下 / bottom-right
        (h_mid, w_mid),              # 中心 / center
    ]

    crops = [image[:, y:y + crop_size, x:x + crop_size] for y, x in coords]
    return torch.stack(crops, dim=0)  # [5, C, S, S]


def twenty_five_crop(image: torch.Tensor, crop_size: int = 224) -> torch.Tensor:
    """25-crop (5×5 grid) / 25-crop evenly-spaced grid.

    在 H/W 维各取 5 个均匀间隔的起点，共 25 个 crop。
    Take 5 evenly-spaced start positions along H and W — 25 crops total.

    比 5-crop 慢 5 倍但通常多 1-2 SRCC 点。
    5× slower than 5-crop, typically +1-2 SRCC points.
    """
    if image.dim() == 4:
        return torch.stack([twenty_five_crop(img, crop_size) for img in image], dim=0)
    if image.dim() != 3:
        raise ValueError(f"期望 3 或 4 维 / expected 3- or 4-D, got {image.dim()}-D")

    c, h, w = image.shape
    assert h >= crop_size and w >= crop_size

    # 5 个起点均匀分布 / 5 evenly-spaced start positions
    ys = torch.linspace(0, h - crop_size, 5).long().tolist()
    xs = torch.linspace(0, w - crop_size, 5).long().tolist()

    crops = [
        image[:, y:y + crop_size, x:x + crop_size]
        for y in ys for x in xs
    ]
    return torch.stack(crops, dim=0)  # [25, C, S, S]


@torch.no_grad()
def tta_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    mode: str = "5crop",
    crop_size: int = 224,
    is_distribution: bool = True,
) -> torch.Tensor:
    """对单张图片做 TTA 推理并平均 / TTA inference with crop averaging.

    Args:
        model:          已 eval 的模型 / model in eval mode
        image:          [C, H, W] 单张图（通常是 resize 到 ~[384, 512]）
                        single image, typically resized to [384, 512]
        mode:           'center' / '5crop' / '25crop'
        crop_size:      crop 尺寸 / crop size
        is_distribution: 模型输出是否为 softmax 分布
                         whether model outputs a softmax distribution

    Returns:
        [num_buckets] 或 [1] 平均后的模型输出 / averaged model output
    """
    if mode == "center":
        # 中心 crop 单次推理 / single center crop
        c, h, w = image.shape
        y = (h - crop_size) // 2
        x = (w - crop_size) // 2
        crop = image[:, y:y + crop_size, x:x + crop_size].unsqueeze(0)
        return model(crop).squeeze(0)

    if mode == "5crop":
        crops = five_crop(image, crop_size)            # [5, C, S, S]
    elif mode == "25crop":
        crops = twenty_five_crop(image, crop_size)     # [25, C, S, S]
    else:
        raise ValueError(f"未知 TTA 模式 / unknown TTA mode: {mode}")

    # 全部 crop 一次 forward / forward all crops at once
    outputs = model(crops)                             # [N, num_buckets] or [N, 1]

    if is_distribution:
        # 分布平均：先 softmax 再平均（model 内部已 softmax 则跳过）
        # For distributions: assume model already outputs softmax, average directly
        return outputs.mean(dim=0)

    # MSE 回归：直接平均标量 / MSE regression: average scalars
    return outputs.mean(dim=0)

"""
XAI 可视化批处理脚本 / XAI visualization batch script.

用途 / Purpose:
    - 从测试集里挑选 N 张图片（跨 MOS 范围），生成 Grad-CAM 或 Attention Rollout
    - 保存为 PNG 网格供报告使用 / save as PNG grid for the paper

    - Picks N test images spanning the MOS range, generates Grad-CAM or
      Attention Rollout, saves as a PNG grid for the report.

用法 / Usage:
    # Grad-CAM (ResNet-50)
    python -m src.xai.visualize \
        --config configs/resnet50_emd.yaml \
        --ckpt outputs/checkpoints/resnet50_emd_best.pt \
        --method gradcam --n-images 16 \
        --output outputs/heatmaps/resnet_gradcam.png

    # Attention Rollout (ViT)
    python -m src.xai.visualize \
        --config configs/vit_lora_emd.yaml \
        --ckpt outputs/checkpoints/vit_b16_lora_emd_best.pt \
        --method rollout --n-images 16 \
        --output outputs/heatmaps/vit_rollout.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ..datasets import build_dataset
from ..models import build_model
from ..utils.distribution import expected_score


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def denormalize_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    """把归一化后的 tensor 还原为 [0,1] 的 RGB 图 / denormalize to [0,1] RGB."""
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)  # CHW → HWC
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0.0, 1.0)


def pick_diverse_samples(ds, n: int, seed: int = 0) -> list:
    """在测试集上挑选分数跨度大的 N 个样本.

    Pick N samples spanning the MOS range — groups by quartile and samples
    evenly so the visualization covers low/mid/high quality examples.
    """
    mos_list = ds.df["MOS"].values
    rng = np.random.default_rng(seed)

    # 按 MOS 分位数分 4 组 / quartile binning
    q = np.quantile(mos_list, [0.25, 0.5, 0.75])
    groups = [
        np.where(mos_list <= q[0])[0],
        np.where((mos_list > q[0]) & (mos_list <= q[1]))[0],
        np.where((mos_list > q[1]) & (mos_list <= q[2]))[0],
        np.where(mos_list > q[2])[0],
    ]

    per_group = max(1, n // 4)
    picks = []
    for g in groups:
        if len(g) > 0:
            picks.extend(rng.choice(g, size=min(per_group, len(g)), replace=False))
    return picks[:n]


def save_grid(
    images: list,
    heatmaps: list,
    titles: list,
    output_path: str,
    cols: int = 4,
) -> None:
    """把 (img, heatmap) 对并排画成网格 / render (img, heatmap) pairs as a grid.

    每个 cell 并排显示原图 + 热力图叠加 / each cell shows orig + overlay side-by-side.
    """
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 3, rows * 6))

    if rows * 2 == 1:
        axes = axes[None, :]

    for i, (img, hm, title) in enumerate(zip(images, heatmaps, titles)):
        r, c = i // cols, i % cols
        # 原图 / original
        axes[r * 2, c].imshow(img)
        axes[r * 2, c].set_title(title, fontsize=8)
        axes[r * 2, c].axis("off")

        # 热力图叠加 / heatmap overlay
        axes[r * 2 + 1, c].imshow(img)
        axes[r * 2 + 1, c].imshow(hm, cmap="jet", alpha=0.5)
        axes[r * 2 + 1, c].axis("off")

    # 清空多余子图 / hide unused cells
    for i in range(n, rows * cols):
        r, c = i // cols, i % cols
        axes[r * 2, c].axis("off")
        axes[r * 2 + 1, c].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[OK] 保存可视化 / saved visualization: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="XAI 可视化 / XAI visualization")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--method", required=True, choices=["gradcam", "rollout"],
        help="gradcam=ResNet / rollout=ViT"
    )
    parser.add_argument("--n-images", type=int, default=16)
    parser.add_argument("--output", required=True, help="PNG 输出路径 / PNG output path")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device

    # 模型 / model
    model = build_model(**cfg["model"]).to(device)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()

    # 数据 / data
    data_cfg = cfg["data"]
    ds = build_dataset(
        data_cfg["dataset"],
        split=args.split,
        root=data_cfg["root"],
        image_size=data_cfg["image_size"],
        resize_to=tuple(data_cfg["resize_to"]),
        num_buckets=data_cfg["num_buckets"],
        min_std=data_cfg["min_std"],
        split_seed=data_cfg["split_seed"],
        split_ratio=tuple(data_cfg["split_ratio"]),
        return_distribution=True,
    )
    indices = pick_diverse_samples(ds, args.n_images, seed=0)
    print(f"[OK] 抽样 / sampled {len(indices)} images spanning MOS quartiles")

    # 按方法准备 / dispatch by method
    if args.method == "gradcam":
        from .gradcam import ResNetGradCAM
        cam = ResNetGradCAM(model, device=device)

        images, heatmaps, titles = [], [], []
        for idx in indices:
            img, _target_dist, mos = ds[idx]
            img_batch = img.unsqueeze(0).to(device)

            # 预测分数 / prediction
            with torch.no_grad():
                prob = model(img_batch)
                pred_score = expected_score(
                    prob, num_buckets=data_cfg["num_buckets"]
                ).item()

            hm = cam(img_batch)  # [H, W] in [0,1]
            images.append(denormalize_to_rgb(img))
            heatmaps.append(hm)
            titles.append(f"GT={mos.item():.1f} / Pred={pred_score:.1f}")

        save_grid(images, heatmaps, titles, args.output)

    elif args.method == "rollout":
        from .attention_rollout import register_attention_hooks, attention_rollout, rollout_heatmap_to_image_size

        images, heatmaps, titles = [], [], []
        for idx in indices:
            img, _target_dist, mos = ds[idx]
            img_batch = img.unsqueeze(0).to(device)

            # 注册 hooks，forward，收集注意力
            # Register hooks, forward, collect attentions
            attn_storage, hooks = register_attention_hooks(model)
            try:
                with torch.no_grad():
                    prob = model(img_batch)
                    pred_score = expected_score(
                        prob, num_buckets=data_cfg["num_buckets"]
                    ).item()
            finally:
                for h in hooks:
                    h.remove()

            if not attn_storage:
                # timm ViT 默认不输出 attn weights，需 patch
                # timm ViT doesn't return attn weights by default — skip with warning
                print(f"[警告 / WARN] idx={idx}: 未捕获到注意力 / no attention captured")
                continue

            rollout = attention_rollout(attn_storage)         # [1, 14, 14]
            hm = rollout_heatmap_to_image_size(
                rollout, image_size=data_cfg["image_size"]
            )[0].cpu().numpy()

            images.append(denormalize_to_rgb(img))
            heatmaps.append(hm)
            titles.append(f"GT={mos.item():.1f} / Pred={pred_score:.1f}")

        if images:
            save_grid(images, heatmaps, titles, args.output)
        else:
            print("[错误 / ERROR] 没有可视化样本 / no visualization samples")


if __name__ == "__main__":
    main()

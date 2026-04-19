"""
评估入口：支持 5-crop / 25-crop TTA 与跨数据集评估.

Evaluation entry: supports 5-crop / 25-crop TTA and cross-dataset eval.

用法 / Usage:
    # KonIQ-10k 测试集（in-dataset）/ in-dataset eval
    python -m src.evaluate \
        --config configs/resnet50_emd.yaml \
        --ckpt outputs/checkpoints/resnet50_emd_best.pt \
        --split test --tta 5crop

    # 跨数据集（SPAQ）/ cross-dataset
    python -m src.evaluate \
        --config configs/resnet50_emd.yaml \
        --ckpt outputs/checkpoints/resnet50_emd_best.pt \
        --cross-dataset spaq --cross-root data/spaq/

参见技术文档 Section 8 / See spec Section 8.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import build_dataset
from .models import build_model
from .utils.distribution import expected_score
from .utils.metrics import compute_all_metrics
from .utils.multi_crop import tta_inference


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup() -> None:
    """与训练一致的确定性与精度设置 / same determinism & precision as training."""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True


def load_model_from_ckpt(cfg: dict, ckpt_path: str, device: str):
    """根据 config + ckpt 加载模型 / build model and load weights.

    注 / Note:
        PyTorch 2.6+ 默认 weights_only=True，会拒绝包含 numpy 标量、自定义
        pickle 对象的 ckpt。我们自己保存的 ckpt 可信，显式设 False。
        PyTorch 2.6+ defaults weights_only=True, which rejects checkpoints
        containing numpy scalars or custom pickled objects. Our own ckpts
        are trusted — explicitly pass False.
    """
    model = build_model(**cfg["model"]).to(device)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # 兼容训练脚本保存的完整 dict / accept the full dict we save in train.py
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    # 兼容 torch.compile 前缀 / strip torch.compile prefix if present
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


@torch.no_grad()
def evaluate_with_tta(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    distribution_mode: bool,
    num_buckets: int,
    tta_mode: str,
    image_size: int,
) -> Dict[str, float]:
    """对整个 loader 做推理（含 TTA），返回指标.

    Run inference (incl. TTA) over the entire loader and return metrics.

    Args:
        tta_mode: 'center' / '5crop' / '25crop'
    """
    preds, targets = [], []

    # TTA 要求逐样本处理（因每张图要裁出多个 crop）
    # TTA processes per-sample (multiple crops per image)
    for batch in tqdm(loader, desc=f"[eval {tta_mode}]"):
        # 解包：训练集返回 3 元组，SPAQ 返回 2 元组
        # Unpack: training returns 3-tuple, SPAQ returns 2-tuple
        if len(batch) == 3:
            images, _dist, mos = batch
        else:
            images, mos = batch

        images = images.to(device, non_blocking=True)

        if tta_mode == "center":
            # 中心 crop 批处理 / center-crop batched
            out = model(images)
            if distribution_mode:
                score = expected_score(out, num_buckets=num_buckets, max_score=100.0)
            else:
                score = out.squeeze(-1) * 100.0
            preds.append(score.cpu())
        else:
            # 5/25-crop：逐样本 / per-sample TTA
            # 这里 images 已经是 resize 过的，大小足够切 224×224
            # images are already resized — sized to fit 224x224 crops
            batch_scores = []
            for i in range(images.size(0)):
                avg_out = tta_inference(
                    model, images[i],
                    mode=tta_mode,
                    crop_size=image_size,
                    is_distribution=distribution_mode,
                )
                if distribution_mode:
                    score = expected_score(
                        avg_out.unsqueeze(0),
                        num_buckets=num_buckets,
                        max_score=100.0,
                    ).squeeze(0)
                else:
                    score = avg_out.squeeze(-1) * 100.0
                batch_scores.append(score)
            preds.append(torch.stack(batch_scores).cpu())

        targets.append(mos.float())

    pred_all = torch.cat(preds)
    target_all = torch.cat(targets)
    return compute_all_metrics(pred_all, target_all)


def main() -> None:
    parser = argparse.ArgumentParser(description="IQA 评估 / IQA evaluation")
    parser.add_argument("--config", required=True, help="训练时用的 YAML / training YAML")
    parser.add_argument("--ckpt", required=True, help="checkpoint .pt 路径 / ckpt path")
    parser.add_argument(
        "--split", default="test", choices=["val", "test"],
        help="KonIQ 划分 / KonIQ split"
    )
    parser.add_argument(
        "--tta", default=None,
        choices=["center", "5crop", "25crop"],
        help="TTA 模式 / TTA mode (默认读 config / default from config)"
    )
    parser.add_argument(
        "--cross-dataset", default=None, choices=[None, "spaq"],
        help="跨数据集评估 / cross-dataset eval"
    )
    parser.add_argument(
        "--cross-root", default=None, help="跨数据集根目录 / cross-dataset root"
    )
    args = parser.parse_args()

    # Windows 多进程保护 / Windows multi-processing guard
    mp.set_start_method("spawn", force=True)
    setup()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模型 / model
    model = load_model_from_ckpt(cfg, args.ckpt, device)

    # 数据 / data
    data_cfg = cfg["data"]
    image_size = data_cfg["image_size"]
    distribution_mode = cfg["model"].get("distribution_mode", True)
    num_buckets = cfg["model"]["num_buckets"]
    tta_mode = args.tta or cfg["eval"].get("tta", "5crop")

    if args.cross_dataset:
        # 跨数据集评估 / cross-dataset eval
        if not args.cross_root:
            raise ValueError("--cross-dataset 需搭配 --cross-root / needs --cross-root")
        ds = build_dataset(
            args.cross_dataset,
            split="test",
            root=args.cross_root,
            image_size=image_size,
            resize_to=tuple(data_cfg["resize_to"]),
        )
        eval_name = f"{args.cross_dataset} (cross-dataset)"
    else:
        # 内部评估 / in-dataset eval
        ds = build_dataset(
            data_cfg["dataset"],
            split=args.split,
            root=data_cfg["root"],
            image_size=image_size,
            resize_to=tuple(data_cfg["resize_to"]),
            num_buckets=num_buckets,
            min_std=data_cfg["min_std"],
            split_seed=data_cfg["split_seed"],
            split_ratio=tuple(data_cfg["split_ratio"]),
            return_distribution=distribution_mode,
        )
        eval_name = f"{data_cfg['dataset']} / {args.split}"

    # TTA 推理时每张图要出多个 crop，batch 不宜太大
    # Each image produces multiple crops for TTA, keep batch modest
    loader = DataLoader(
        ds,
        batch_size=16 if tta_mode != "center" else 64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    metrics = evaluate_with_tta(
        model, loader, device,
        distribution_mode=distribution_mode,
        num_buckets=num_buckets,
        tta_mode=tta_mode,
        image_size=image_size,
    )

    # 打印结果 / print results
    print("\n" + "=" * 60)
    print(f"评估 / Eval: {eval_name}  |  TTA: {tta_mode}")
    print(f"ckpt: {args.ckpt}")
    print("-" * 60)
    print(f"  SRCC: {metrics['srcc']:.4f}")
    print(f"  PLCC: {metrics['plcc']:.4f}")
    print(f"  KRCC: {metrics['krcc']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}  (normalized)")
    print("=" * 60)


if __name__ == "__main__":
    main()

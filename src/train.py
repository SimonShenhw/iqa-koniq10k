"""
训练入口 / Training entry point.

用法 / Usage:
    python -m src.train --config configs/resnet50_emd.yaml

关键设计 / Key design points:
    1. Windows spawn 修复：多进程 + persistent_workers 的内存泄漏防御
       Windows spawn fix: prevents DataLoader memory leak
    2. TF32 + bf16 AMP 精度栈：激活 Blackwell Tensor Core
       TF32 + bf16 AMP stack: unleashes Blackwell Tensor Cores
    3. torch.compile：默认开启，ViT 可在 config 关闭
       torch.compile: default on; config can disable for ViT
    4. 两阶段冻结（仅 ResNet-50）/ two-phase freezing (ResNet-50 only)
    5. 每 epoch 保存 + 仅保存 best-SRCC / per-epoch + best-SRCC checkpoints

参见技术文档 Section 3、7 / See spec Sections 3 and 7.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import build_dataset
from .losses import EMDLoss
from .models import build_model
from .utils.distribution import expected_score
from .utils.metrics import compute_all_metrics


# ======================================================================
# 工具函数 / Helpers
# ======================================================================
def load_config(path: str) -> Dict[str, Any]:
    """读取 YAML 配置 / load YAML config."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker 确定性种子 / deterministic per-worker seed.

    若不设置，所有 worker 会产生相同的 RandomResizedCrop 结果——
    这会静默地毁掉数据增强。

    Without this, all workers generate identical RandomResizedCrop outputs —
    silently breaking data augmentation.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_determinism(seed: int) -> None:
    """统一设置随机种子 / seed everything."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False  # 速度优先 / speed over strict repro
    torch.backends.cudnn.benchmark = True       # 固定形状下自动选 conv 算法 / autotune


def setup_blackwell_precision() -> None:
    """激活 Blackwell Tensor Core 的 TF32 + AMP 组合.

    Unleash Blackwell Tensor Cores via TF32 + bf16 AMP.

    注意 / Note:
        'high' 在此处启用 TF32（10-bit 尾数，~0.001 相对误差），
        搭配 bf16 autocast 后训练吞吐约翻 2.5×。
        'high' enables TF32 — paired with bf16 autocast gives ~2.5x throughput.
    """
    # FP32 matmul 走 TF32 / FP32 matmul via TF32 Tensor Core
    torch.set_float32_matmul_precision("high")
    # 卷积同样允许 TF32 / conv also allows TF32
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True  # legacy API safety


# ======================================================================
# 数据 / Data
# ======================================================================
def build_dataloaders(cfg: Dict[str, Any]):
    """根据配置构建 train/val DataLoader / build train/val DataLoader."""
    data_cfg = cfg["data"]
    dl_cfg = cfg["dataloader"]

    # 是否以分布形式返回标签 / return distribution labels?
    distribution_mode = cfg["model"].get("distribution_mode", True)

    common_kwargs = dict(
        root=data_cfg["root"],
        image_size=data_cfg["image_size"],
        resize_to=tuple(data_cfg["resize_to"]),
        num_buckets=data_cfg["num_buckets"],
        min_std=data_cfg["min_std"],
        split_seed=data_cfg["split_seed"],
        split_ratio=tuple(data_cfg["split_ratio"]),
        return_distribution=distribution_mode,
    )

    train_ds = build_dataset(data_cfg["dataset"], split="train", **common_kwargs)
    val_ds = build_dataset(data_cfg["dataset"], split="val", **common_kwargs)

    # 固定种子的 generator 确保 shuffle 可复现
    # Fixed-seed generator makes shuffle reproducible
    g = torch.Generator()
    g.manual_seed(cfg["experiment"]["seed"])

    train_loader = DataLoader(
        train_ds,
        batch_size=dl_cfg["train_batch_size"],
        shuffle=True,
        num_workers=dl_cfg["num_workers"],
        pin_memory=dl_cfg["pin_memory"],
        persistent_workers=dl_cfg["persistent_workers"],
        prefetch_factor=dl_cfg["prefetch_factor"],
        drop_last=dl_cfg["drop_last"],
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=dl_cfg["val_batch_size"],
        shuffle=False,
        num_workers=max(dl_cfg["num_workers"] // 2, 2),
        pin_memory=dl_cfg["pin_memory"],
        persistent_workers=dl_cfg["persistent_workers"],
    )
    return train_loader, val_loader


# ======================================================================
# 优化器 / Scheduler
# ======================================================================
def build_optimizer_and_scheduler(
    model: nn.Module,
    cfg: Dict[str, Any],
    steps_per_epoch: int,
):
    """构建 AdamW + 余弦退火 / build AdamW + cosine schedule.

    ResNet-50 支持双学习率组；其他模型统一学习率。
    ResNet-50 uses dual-LR param groups; other models use a single LR.
    """
    train_cfg = cfg["train"]
    wd = train_cfg["weight_decay"]
    lr_head = train_cfg.get("lr_head", train_cfg.get("lr", 1e-4))
    lr_backbone = train_cfg.get("lr_backbone", lr_head)

    # 双 LR 参数组（ResNet-50）/ dual-LR param groups (ResNet-50)
    if hasattr(model, "trainable_parameter_groups"):
        param_groups = model.trainable_parameter_groups(lr_head, lr_backbone)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)
    else:
        # 默认单 LR：所有 requires_grad 的参数
        # Default single-LR: all requires_grad params
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr_head, weight_decay=wd)

    # 余弦退火（按 step 计）+ 可选 warmup
    # Cosine schedule (per-step) + optional warmup
    total_steps = train_cfg["epochs"] * steps_per_epoch
    warmup_steps = train_cfg.get("warmup_epochs", 0) * steps_per_epoch

    if warmup_steps > 0:
        # 线性 warmup + 余弦退火 / linear warmup + cosine
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        # 纯余弦退火 / plain cosine
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps
        )

    return optimizer, scheduler


# ======================================================================
# 单 epoch 训练 / Train one epoch
# ======================================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: str,
    grad_clip: float,
    precision: str,
    distribution_mode: bool,
    epoch: int,
) -> float:
    """训练一个 epoch，返回平均损失 / train one epoch, return mean loss."""
    model.train()
    running_loss, n = 0.0, 0

    # bf16 AMP 上下文 / bf16 AMP context
    autocast_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[precision]
    use_autocast = precision in ("bfloat16", "float16")

    pbar = tqdm(loader, desc=f"[train epoch {epoch}]", leave=False)
    for batch in pbar:
        if distribution_mode:
            images, target_dist, _mos = batch
            target = target_dist.to(device, non_blocking=True)
        else:
            images, mos_norm = batch
            target = mos_norm.to(device, non_blocking=True)

        images = images.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # autocast：conv/matmul 走 bf16，LayerNorm/Softmax/loss 留 FP32
        # autocast: conv/matmul in bf16, LayerNorm/Softmax/loss stay FP32
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast):
            pred = model(images)
            if distribution_mode:
                loss = loss_fn(pred, target)
            else:
                # MSE 回归：pred [B, 1] vs target [B]
                # MSE: pred [B,1] vs target [B]
                loss = F.mse_loss(pred.squeeze(-1), target)

        loss.backward()

        # 梯度裁剪 / grad clipping
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        # 累计损失 / accumulate
        bs = images.size(0)
        running_loss += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(n, 1)


# ======================================================================
# 验证 / Validation
# ======================================================================
@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    distribution_mode: bool,
    num_buckets: int,
) -> Dict[str, float]:
    """在验证集上计算 SRCC/PLCC/RMSE/KRCC / compute val metrics."""
    model.eval()
    preds, targets = [], []

    for batch in tqdm(loader, desc="[val]", leave=False):
        if distribution_mode:
            images, _target_dist, mos = batch
            images = images.to(device, non_blocking=True)
            prob = model(images)  # [B, num_buckets]
            score = expected_score(prob, num_buckets=num_buckets, max_score=100.0)
            preds.append(score.cpu())
            targets.append(mos.cpu())
        else:
            images, mos_norm = batch
            images = images.to(device, non_blocking=True)
            # MSE 模式：输出归一化 → 还原到 [0, 100]
            # MSE mode: normalized output → rescale to [0, 100]
            pred_norm = model(images).squeeze(-1)
            preds.append((pred_norm * 100.0).cpu())
            targets.append((mos_norm * 100.0).cpu())

    pred_all = torch.cat(preds)
    target_all = torch.cat(targets)
    return compute_all_metrics(pred_all, target_all)


# ======================================================================
# 主训练流程 / Main train loop
# ======================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="IQA 训练 / IQA training")
    parser.add_argument("--config", required=True, help="YAML 配置 / YAML config path")
    parser.add_argument("--resume", default=None, help="续训 ckpt 路径 / resume ckpt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_cfg = cfg["experiment"]
    train_cfg = cfg["train"]

    # ---- 0. Windows spawn 防御：必须在任何 CUDA/DataLoader 之前
    # ---- 0. Windows spawn guard: MUST run before any CUDA/DataLoader
    mp.set_start_method("spawn", force=True)

    # 确定性 + 精度栈 / determinism + precision stack
    setup_determinism(exp_cfg["seed"])
    setup_blackwell_precision()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[警告 / WARN] 未检测到 GPU — 训练将极慢 / no GPU detected, will be slow")

    # ---- 1. 数据 / data ----
    train_loader, val_loader = build_dataloaders(cfg)
    steps_per_epoch = len(train_loader)

    # ---- 2. 模型 / model ----
    model = build_model(**cfg["model"]).to(device)
    distribution_mode = cfg["model"].get("distribution_mode", True)
    num_buckets = cfg["model"]["num_buckets"]

    # Phase 1 冻结（仅 ResNet-50）/ Phase 1 freeze (ResNet-50 only)
    freeze_cfg = train_cfg.get("freeze_schedule")
    if freeze_cfg and hasattr(model, "freeze_backbone_except_fc"):
        model.freeze_backbone_except_fc()
        print("[Phase 1] 冻结主干 / backbone frozen; head-only training")

    # ViT LoRA：打印可训练比例 / print trainable ratio
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    # ---- 3. 优化器 + 调度 / optimizer + scheduler ----
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, steps_per_epoch)

    # ---- 4. 损失 / loss ----
    if train_cfg["loss"] == "emd":
        loss_fn = EMDLoss(r=2).to(device)
    elif train_cfg["loss"] == "mse":
        loss_fn = nn.MSELoss().to(device)  # 不会真的用，见 train_one_epoch
    else:
        raise ValueError(f"未知 loss / unknown loss: {train_cfg['loss']}")

    # ---- 5. torch.compile（可选）/ torch.compile (optional) ----
    # Windows + Python 3.14 + Triton 缺失 → torch.compile 会在首次 forward 崩溃。
    # 这里做双重防御：1) 启动前检测 Triton；2) 构造失败时 fallback eager。
    # Windows + Python 3.14 lacks Triton wheels → torch.compile crashes at
    # first forward. Double guard: 1) check Triton availability; 2) fall back
    # on eager if construction fails. Matches spec Section 12 risk mitigation.
    if train_cfg.get("compile", False):
        try:
            import importlib.util
            if importlib.util.find_spec("triton") is None:
                print("[提示 / INFO] Triton 未安装，回退 eager 模式 / no Triton, using eager")
            else:
                model = torch.compile(model, mode="default")
                print("[OK] torch.compile 已启用 / compile enabled")
        except Exception as e:
            print(f"[警告] torch.compile 失败，回退 eager / failed, fallback eager: {e}")

    # ---- 6. 续训 / resume ----
    start_epoch = 0
    best_metric = -1.0
    if args.resume:
        # 自己的 ckpt，trusted / our own ckpt, trusted
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best_metric = state.get("best_metric", -1.0)
        print(f"[OK] 续训自 epoch {start_epoch} / resumed from epoch {start_epoch}")

    # ---- 7. 训练循环 / training loop ----
    output_dir = Path(exp_cfg["output_dir"]) / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_metric_name = cfg["eval"]["save_best_metric"]
    patience = train_cfg.get("early_stop_patience", 5)
    patience_counter = 0

    for epoch in range(start_epoch, train_cfg["epochs"]):
        t0 = time.time()

        # Phase 2 解冻 / Phase 2 unfreeze
        if (
            freeze_cfg
            and epoch == freeze_cfg["phase1_epochs"]
            and hasattr(model, "unfreeze_layers")
        ):
            model.unfreeze_layers(freeze_cfg["unfreeze_layers"])
            # 解冻后需重建 optimizer/scheduler，新参数纳入优化
            # Rebuild optimizer/scheduler to include newly-trainable params
            optimizer, scheduler = build_optimizer_and_scheduler(
                model, cfg, steps_per_epoch
            )
            print(f"[Phase 2] 已解冻 / unfroze: {freeze_cfg['unfreeze_layers']}")

        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scheduler,
            device=device,
            grad_clip=train_cfg.get("grad_clip", 0),
            precision=train_cfg.get("precision", "bfloat16"),
            distribution_mode=distribution_mode,
            epoch=epoch,
        )

        metrics = validate(model, val_loader, device, distribution_mode, num_buckets)
        dt = time.time() - t0

        print(
            f"[epoch {epoch:3d}] train_loss={train_loss:.4f} "
            f"val_srcc={metrics['srcc']:.4f} val_plcc={metrics['plcc']:.4f} "
            f"val_rmse={metrics['rmse']:.4f} time={dt:.1f}s"
        )

        # 保存 / save checkpoints
        state_to_save = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "best_metric": best_metric,
            "config": cfg,
        }
        torch.save(state_to_save, output_dir / f"{exp_cfg['name']}_last.pt")

        cur_metric = metrics[best_metric_name]
        if cur_metric > best_metric:
            best_metric = cur_metric
            state_to_save["best_metric"] = best_metric
            torch.save(state_to_save, output_dir / f"{exp_cfg['name']}_best.pt")
            patience_counter = 0
            print(f"  → 新最佳 / new best {best_metric_name}: {best_metric:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[提前停止 / Early stop] patience={patience} 已耗尽")
                break

    print(f"\n训练完成 / Training done. Best {best_metric_name}: {best_metric:.4f}")


if __name__ == "__main__":
    # Windows 下必须用 __main__ 保护 / required guard on Windows
    main()

"""
KonIQ-10k 数据集加载 / KonIQ-10k dataset loader.

约定的目录结构 / Expected directory layout:
    data/koniq10k/
        ├── 1024x768/              # 原始高清图 / original images
        │   ├── img_1.jpg
        │   └── ...
        └── koniq10k_scores_and_distributions.csv
             # 包含字段 / contains columns:
             #   image_name, MOS, SD, c1..c5 (per-bucket counts)

划分规则 / Split rules:
- 基于 image_id 确定性划分（种子=42），避免重复图片泄漏
  image_id-based deterministic split (seed=42), avoids duplicate-image leakage
- 默认 8:1:1 / default 8:1:1 train/val/test

参见技术文档 Section 3 / See spec Section 3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..utils.distribution import generate_gaussian_target


# ImageNet 标准化常数（所有 IQA 预训练主干都使用）
# ImageNet normalization constants — used by every pretrained IQA backbone
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _find_image_dir(root: Path) -> Path:
    """自动探测图像所在子目录 / Auto-detect which subfolder holds the images."""
    # KonIQ 镜像的常见布局 / common mirror layouts
    for sub in ["1024x768", "512x384", "images", ""]:
        d = root / sub if sub else root
        if d.is_dir() and any(d.glob("*.jpg")):
            return d
    raise FileNotFoundError(
        f"在 {root} 下未找到 *.jpg / no *.jpg under {root} — 请确认解压路径"
    )


def _find_csv(root: Path) -> Path:
    """定位评分 CSV / Locate the scores CSV."""
    # 不同镜像的命名差异 / name differs across mirrors
    for name in [
        "koniq10k_scores_and_distributions.csv",
        "koniq10k_distributions_sets.csv",
        "koniq10k_scores.csv",
    ]:
        p = root / name
        if p.is_file():
            return p
    # 回退：任何 koniq 相关的 csv / fallback: any koniq-related csv
    candidates = list(root.glob("*.csv"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"在 {root} 下未找到 CSV / no CSV under {root}")


def _deterministic_split(
    ids: np.ndarray,
    ratio: Tuple[float, float, float],
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """基于 image_id 的确定性划分 / image_id-based deterministic split.

    先去重（防止同图多行被错分），再打乱，再切分。
    Deduplicate first (prevents same image landing in both sides),
    then shuffle deterministically, then slice.
    """
    unique_ids = np.unique(ids)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_ids)

    n = len(shuffled)
    n_train = int(n * ratio[0])
    n_val = int(n * ratio[1])

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train:n_train + n_val]
    test_ids = shuffled[n_train + n_val:]
    return train_ids, val_ids, test_ids


class KonIQ10kDataset(Dataset):
    """KonIQ-10k 数据集 / KonIQ-10k dataset.

    Args:
        root:          数据集根目录 / dataset root
        split:         'train' | 'val' | 'test'
        image_size:    训练裁剪/推理 crop 尺寸 / crop size
        resize_to:     预处理 resize 的 (H, W) / pre-resize (H, W)
        num_buckets:   高斯分布桶数 / number of Gaussian bins
        min_std:       Gaussian std 下限（桶单位）/ std floor in bucket units
        split_seed:    划分随机种子 / split seed
        split_ratio:   (train, val, test) 比例 / split ratios
        return_distribution:
                       True 返回分布标签，False 仅返回标量 MOS
                       True → return distribution label, False → scalar MOS only
        transform:     自定义 transform，None 时使用默认
                       custom transform; uses default if None
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 224,
        resize_to: Tuple[int, int] = (384, 512),
        num_buckets: int = 10,
        min_std: float = 0.5,
        split_seed: int = 42,
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        return_distribution: bool = True,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.num_buckets = num_buckets
        self.min_std = min_std
        self.return_distribution = return_distribution

        # 探测数据路径 / locate data
        self.image_dir = _find_image_dir(self.root)
        csv_path = _find_csv(self.root)

        # 读 CSV，适配不同镜像的列名
        # Load CSV, robust to different mirror column names
        df = pd.read_csv(csv_path)
        df = self._normalize_columns(df)

        # 划分 / split
        train_ids, val_ids, test_ids = _deterministic_split(
            df["image_name"].values, split_ratio, split_seed
        )
        id_map = {"train": train_ids, "val": val_ids, "test": test_ids}
        if split not in id_map:
            raise ValueError(f"split 必须是 train/val/test / got {split}")
        self.df = df[df["image_name"].isin(id_map[split])].reset_index(drop=True)

        # Transform 构建 / build transforms
        self.transform = transform or self._default_transform(split, image_size, resize_to)

    # ---------------------------------------------------------------
    # 列名归一化 / column-name normalization
    # ---------------------------------------------------------------
    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """把不同镜像的列名对齐为标准集合，并统一到 [0, 100] 量表.

        Unify column names across mirrors AND rescale MOS to [0, 100].

        关于 KonIQ 的 MOS 量表 / About KonIQ's MOS scale:
            - 官方 CSV 里 `MOS` 是 5 点李克特平均（范围 1-5）
            - `MOS_zscore` 是 z-score 重新映射到 [0, 100]，论文报告用的是这个
            - 若仅有原始 `MOS` 列，需要线性映射 5→100、1→0 才能喂进 distribution.py

            - Official CSV: `MOS` is a 1-5 Likert average
            - `MOS_zscore` is z-score rescaled to [0, 100] — papers use this
            - If only raw `MOS` is present, we linearly map 1→0 and 5→100
        """
        cols_lower = {c.lower(): c for c in df.columns}

        # ---- 1. image_name 列 / image name column ----
        for candidate in ["image_name", "filename", "image", "img_id"]:
            if candidate in cols_lower:
                df = df.rename(columns={cols_lower[candidate]: "image_name"})
                break

        # ---- 2. MOS 列（优先选 [0,100] 量表）/ MOS (prefer 0-100 scale) ----
        # 注意：若同时有 MOS 和 MOS_zscore，先 drop 原始 MOS，再把 zscore 重命名，
        # 避免产生两个同名列。
        # Note: if both MOS and MOS_zscore exist, drop raw MOS first, then
        # rename zscore — otherwise we end up with two columns named "MOS".
        if "mos_zscore" in cols_lower:
            if "MOS" in df.columns:
                df = df.drop(columns=["MOS"])
            df = df.rename(columns={cols_lower["mos_zscore"]: "MOS"})
        else:
            # 回退：原始 MOS / fall back to raw MOS
            for candidate in ["mos", "score"]:
                if candidate in cols_lower and "MOS" not in df.columns:
                    df = df.rename(columns={cols_lower[candidate]: "MOS"})
                    break

            # 若 MOS 仍在 [1, 5] 量表，线性映射到 [0, 100]
            # If MOS is on [1, 5] (Likert), linearly map to [0, 100]
            if "MOS" in df.columns and df["MOS"].max() <= 5.5:
                df["MOS"] = (df["MOS"] - 1.0) / 4.0 * 100.0

        # ---- 3. SD 列 / SD column ----
        for candidate in ["sd", "std", "mos_std", "std_dev"]:
            if candidate in cols_lower and "SD" not in df.columns:
                df = df.rename(columns={cols_lower[candidate]: "SD"})
                break

        # SD 判断：官方 CSV 中 SD 是 1-5 量表下的 std，要放大到 [0,100] 量纲
        # 原始 [1,5] → [0,100] 的线性变换放大因子为 25 (100/4)
        # The official CSV keeps SD on the 1-5 Likert scale; scale it by 25
        # to align with the [0, 100] MOS target used by generate_gaussian_target.
        if "SD" in df.columns and df["SD"].max() < 5.0:
            df["SD"] = df["SD"] * 25.0

        required = {"image_name", "MOS", "SD"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(
                f"CSV 缺少必要列 / CSV missing columns: {missing}. "
                f"现有列 / available: {list(df.columns)}"
            )
        return df

    # ---------------------------------------------------------------
    # Transform / 数据增强
    # ---------------------------------------------------------------
    @staticmethod
    def _default_transform(
        split: str,
        image_size: int,
        resize_to: Tuple[int, int],
    ) -> Callable:
        """默认数据增强 / default transform.

        规范 / Rules:
        - 训练: Resize → RandomResizedCrop(scale=(0.5,1.0)) → HFlip → Normalize
        - 测试: Resize → CenterCrop → Normalize
        - 绝对禁止 / FORBIDDEN: GaussianBlur, ColorJitter, RandomErasing, 任何
                                会改变"图像质量"的增强（会污染标签）。
          anything that changes perceived quality — it would corrupt the label.
        """
        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        resize = transforms.Resize(resize_to)

        if split == "train":
            return transforms.Compose([
                resize,
                # scale=(0.5, 1.0) 是 IQA 专用设置，默认 (0.08, 1.0) 过于激进
                # IQA-specific scale — default (0.08, 1.0) is too aggressive
                transforms.RandomResizedCrop(
                    image_size, scale=(0.5, 1.0), antialias=True
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                normalize,
            ])

        # 验证 / 测试：确定性中心 crop / val/test: deterministic center crop
        return transforms.Compose([
            resize,
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])

    # ---------------------------------------------------------------
    # Dataset 接口 / Dataset interface
    # ---------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.image_dir / row["image_name"]

        # 读取 RGB 图像 / read RGB image
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        mos = torch.tensor(float(row["MOS"]), dtype=torch.float32)
        std = torch.tensor(float(row["SD"]), dtype=torch.float32)

        if not self.return_distribution:
            # MSE 回归模式：返回归一化标量 / MSE mode: return normalized scalar
            return image, mos / 100.0

        # EMD 模式：返回 Gaussian 分布 + 原始 MOS（用于 SRCC）
        # EMD mode: return Gaussian distribution + raw MOS (for SRCC)
        target_dist = generate_gaussian_target(
            mos.unsqueeze(0),
            std.unsqueeze(0),
            num_buckets=self.num_buckets,
            min_std=self.min_std,
        ).squeeze(0)

        return image, target_dist, mos

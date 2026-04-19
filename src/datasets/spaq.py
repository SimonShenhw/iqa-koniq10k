"""
SPAQ 数据集（跨数据集评估专用）/ SPAQ dataset (for cross-dataset evaluation).

只读 / Test-only — 用已在 KonIQ-10k 上训练好的模型做零样本推理。
Used for zero-shot inference with models trained on KonIQ-10k.

约定目录结构 / Expected layout:
    data/spaq/
        ├── TestImage/
        │   └── *.jpg
        └── Annotations/
            └── MOS and Image attribute scores.xlsx
             或 / or: MOSAndAttributes.csv

参见技术文档 Section 3 / See spec Section 3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .koniq10k import IMAGENET_MEAN, IMAGENET_STD


def _find_spaq_image_dir(root: Path) -> Path:
    """探测 SPAQ 图像目录 / find SPAQ image folder.

    兼容多种解压布局 / supports multiple extraction layouts:
        root/TestImage/*.jpg          (直接解压 / flat)
        root/SPAQ/TestImage/*.jpg     (HF tarball 多嵌一层 / HF tarball, nested)
        root/images/*.jpg             (某些镜像)
        root/*.jpg                    (完全扁平)
    """
    candidates = [
        "TestImage", "SPAQ/TestImage",
        "images", "SPAQ/images",
        "",
    ]
    for sub in candidates:
        d = root / sub if sub else root
        if d.is_dir() and any(d.glob("*.jpg")):
            return d
    raise FileNotFoundError(f"未找到 SPAQ 图像 / no SPAQ images in {root}")


def _find_spaq_labels(root: Path) -> Path:
    """探测 SPAQ 标签文件（xlsx 或 csv）/ find SPAQ labels (xlsx/csv)."""
    for pattern in [
        "Annotations/*.xlsx",
        "Annotations/*.csv",
        "SPAQ/Annotations/*.xlsx",   # HF tarball 布局 / HF tarball layout
        "SPAQ/Annotations/*.csv",
        "*.xlsx",
        "*.csv",
    ]:
        hits = list(root.glob(pattern))
        if hits:
            # 偏好含 MOS 的文件（SPAQ 有多个 xlsx）/ prefer file with 'MOS' in name
            for h in hits:
                if "mos" in h.name.lower():
                    return h
            return hits[0]
    raise FileNotFoundError(f"未找到 SPAQ 标签 / no SPAQ labels in {root}")


class SPAQDataset(Dataset):
    """SPAQ — 仅用于跨数据集零样本评估 / SPAQ — zero-shot cross-dataset eval only.

    Args:
        root:        SPAQ 数据根目录 / dataset root
        split:       忽略，SPAQ 全部作为 test / ignored, always test
        image_size:  中心 crop 尺寸 / center crop size
        resize_to:   pre-crop resize (H, W)
        transform:   自定义 transform / custom transform
    """

    def __init__(
        self,
        root: str,
        split: str = "test",
        image_size: int = 224,
        resize_to: Tuple[int, int] = (384, 512),
        transform: Optional[Callable] = None,
        **_unused,  # 兼容 build_dataset 统一接口 / swallow extra kwargs
    ):
        super().__init__()
        self.root = Path(root)
        self.image_dir = _find_spaq_image_dir(self.root)
        label_path = _find_spaq_labels(self.root)

        # 读取标签文件（xlsx 或 csv）/ load labels (xlsx or csv)
        if label_path.suffix.lower() == ".xlsx":
            df = pd.read_excel(label_path)
        else:
            df = pd.read_csv(label_path)

        df = self._normalize_columns(df)
        self.df = df

        # Transform：跨数据集推理 → 中心 crop（确定性）/ cross-dataset inference → center crop
        self.transform = transform or transforms.Compose([
            transforms.Resize(resize_to),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """统一列名 / normalize column names."""
        cols_lower = {c.lower(): c for c in df.columns}

        # 图像名列 / image name column
        for candidate in ["image name", "image_name", "filename", "image"]:
            if candidate in cols_lower:
                df = df.rename(columns={cols_lower[candidate]: "image_name"})
                break

        # MOS 列 / MOS column
        for candidate in ["mos", "quality_mos", "overall"]:
            if candidate in cols_lower and "MOS" not in df.columns:
                df = df.rename(columns={cols_lower[candidate]: "MOS"})
                break

        if "image_name" not in df.columns or "MOS" not in df.columns:
            raise KeyError(
                f"SPAQ 标签缺少 image_name 或 MOS 列 / missing cols. "
                f"现有 / available: {list(df.columns)}"
            )
        return df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.image_dir / row["image_name"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        mos = torch.tensor(float(row["MOS"]), dtype=torch.float32)
        # SPAQ 没有可靠的 per-image std，跨库评估只需 MOS
        # SPAQ lacks reliable per-image std; cross-dataset eval needs MOS only
        return image, mos

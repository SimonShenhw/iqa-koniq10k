"""数据集模块 / Datasets module.

提供 / Exposes:
- KonIQ10kDataset: 主训练集 / primary training set
- SPAQDataset: 跨数据集评估 / cross-dataset evaluation
- build_dataset: 工厂函数 / factory dispatcher
"""

from __future__ import annotations

from .koniq10k import KonIQ10kDataset
from .spaq import SPAQDataset


def build_dataset(name: str, split: str, **kwargs):
    """根据名称分派数据集构造 / Dispatch dataset construction by name.

    Args:
        name: 数据集名称 / dataset name (e.g. 'koniq10k', 'spaq')
        split: 数据划分 / split name (train/val/test)
        **kwargs: 传递给具体数据集类 / passed to dataset class
    """
    name = name.lower()
    if name == "koniq10k":
        return KonIQ10kDataset(split=split, **kwargs)
    if name == "spaq":
        return SPAQDataset(split=split, **kwargs)
    raise ValueError(f"未知数据集 / Unknown dataset: {name}")


__all__ = ["KonIQ10kDataset", "SPAQDataset", "build_dataset"]

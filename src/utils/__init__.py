"""通用工具模块 / Utility module.

提供 / Exposes:
- generate_gaussian_target / expected_score: MOS ↔ 分布转换
- compute_srcc / compute_plcc / compute_rmse / compute_krcc: 评估指标
- five_crop / twenty_five_crop: 测试时增强 (TTA)
"""

from .distribution import expected_score, generate_gaussian_target
from .metrics import compute_krcc, compute_plcc, compute_rmse, compute_srcc
from .multi_crop import five_crop, twenty_five_crop

__all__ = [
    "generate_gaussian_target",
    "expected_score",
    "compute_srcc",
    "compute_plcc",
    "compute_rmse",
    "compute_krcc",
    "five_crop",
    "twenty_five_crop",
]

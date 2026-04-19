"""
IQA 评估指标 / IQA evaluation metrics.

SRCC = Spearman Rank Correlation — 主指标，对单调变换鲁棒
PLCC = Pearson Linear Correlation — 线性相关
RMSE = Root Mean Square Error — 归一化后均方根误差
KRCC = Kendall Rank Correlation — 补充指标

参见技术文档 Section 8 / See technical spec Section 8.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch
from scipy import stats

# 支持 Tensor 或 ndarray / accept both Tensor and ndarray
ArrayLike = Union[torch.Tensor, np.ndarray]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    """统一转成一维 numpy 数组 / unify to 1-D numpy array."""
    if isinstance(x, torch.Tensor):
        # 转到 CPU，detach，flatten / move to CPU, detach, flatten
        return x.detach().cpu().numpy().reshape(-1)
    return np.asarray(x).reshape(-1)


def compute_srcc(pred: ArrayLike, target: ArrayLike) -> float:
    """Spearman 秩相关 / Spearman rank correlation.

    主 IQA 指标。对任意单调变换（含非线性校准）都不变。
    Primary IQA metric. Invariant to any monotonic transform including
    nonlinear calibration — preferred when we care about ordering.
    """
    pred_np, target_np = _to_numpy(pred), _to_numpy(target)
    srcc, _ = stats.spearmanr(pred_np, target_np)
    return float(srcc)


def compute_plcc(pred: ArrayLike, target: ArrayLike) -> float:
    """Pearson 线性相关 / Pearson linear correlation.

    衡量线性关系强度，通常在非线性校准（如 4-param logistic）后报告。
    Measures linear relationship. In IQA literature, PLCC is often computed
    AFTER a nonlinear (e.g. 4-parameter logistic) calibration; here we give
    the raw PLCC — apply calibration externally if needed.
    """
    pred_np, target_np = _to_numpy(pred), _to_numpy(target)
    plcc, _ = stats.pearsonr(pred_np, target_np)
    return float(plcc)


def compute_rmse(pred: ArrayLike, target: ArrayLike, normalize: bool = True) -> float:
    """归一化 RMSE / Normalized RMSE.

    Args:
        pred / target: 预测与真实分数 / predictions and ground truth
        normalize:   若 True，先除以 100 归一到 [0,1] 再算 RMSE
                     if True, divide by 100 then compute — match spec scale

    Returns:
        标量 RMSE / scalar RMSE
    """
    pred_np, target_np = _to_numpy(pred), _to_numpy(target)
    if normalize:
        pred_np = pred_np / 100.0
        target_np = target_np / 100.0
    return float(np.sqrt(np.mean((pred_np - target_np) ** 2)))


def compute_krcc(pred: ArrayLike, target: ArrayLike) -> float:
    """Kendall 秩相关 / Kendall rank correlation.

    补充指标；对样本量敏感但对异常值更鲁棒。
    Supplementary metric; more sensitive to sample size but robust to outliers.
    """
    pred_np, target_np = _to_numpy(pred), _to_numpy(target)
    # scipy ≥ 1.11 默认 'b' 方法 / default 'b' variant
    krcc, _ = stats.kendalltau(pred_np, target_np)
    return float(krcc)


def compute_all_metrics(pred: ArrayLike, target: ArrayLike) -> dict:
    """一次返回所有指标 / Compute all metrics at once."""
    return {
        "srcc": compute_srcc(pred, target),
        "plcc": compute_plcc(pred, target),
        "rmse": compute_rmse(pred, target, normalize=True),
        "krcc": compute_krcc(pred, target),
    }

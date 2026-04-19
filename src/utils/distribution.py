"""
MOS ↔ 概率分布转换 / MOS ↔ probability distribution conversion.

IQA 项目中，KonIQ-10k 的原始标签是 MOS (mean opinion score) + std。
要训练 EMD 损失，必须把 (mos, std) 映射成 num_buckets 维的离散概率分布。
反过来评估时，用分布的期望值还原回连续分数，再算 SRCC/PLCC。

In this IQA project, KonIQ-10k provides raw labels as MOS (mean opinion
score) + std. To train with EMD loss, we convert (mos, std) into a discrete
probability distribution over `num_buckets` bins. At evaluation, we recover
a scalar score via the distribution's expected value and compute SRCC/PLCC.

参见技术文档 Section 4 / See technical spec Section 4.
"""

from __future__ import annotations

import torch


def generate_gaussian_target(
    mos: torch.Tensor,
    std: torch.Tensor,
    num_buckets: int = 10,
    max_score: float = 100.0,
    min_std: float = 0.5,
) -> torch.Tensor:
    """从 (MOS, std) 构造离散 Gaussian 概率分布 / Build Gaussian-shaped discrete PDF.

    Args:
        mos:        [B] 原始 MOS，取值范围 [0, max_score] / raw MOS in [0, max_score]
        std:        [B] 原始标准差 / raw std
        num_buckets: 桶数（默认 10）/ number of bins (default 10)
        max_score:  MOS 的最大可能值 / upper bound of MOS
        min_std:    桶单位下的最小 std，避免退化为 one-hot
                    minimum std in bucket units — prevents one-hot collapse

    Returns:
        [B, num_buckets] 每行归一化概率分布 / row-normalized PDF
    """
    # 将 MOS 和 std 都从原始尺度归一化到桶尺度
    # Rescale MOS and std from raw range to bucket-index range
    scaled_mos = (mos / max_score) * num_buckets
    scaled_std = torch.clamp(std / max_score * num_buckets, min=min_std)

    # 桶中心坐标：1, 2, ..., num_buckets / bin centers: 1..num_buckets
    classes = torch.arange(
        1, num_buckets + 1, device=mos.device, dtype=torch.float32
    )

    # 广播：[B, 1] 与 [num_buckets] → [B, num_buckets]
    # Broadcasting: [B,1] × [num_buckets] → [B, num_buckets]
    scaled_mos = scaled_mos.unsqueeze(1)
    scaled_std = scaled_std.unsqueeze(1)

    # 高斯核 / Gaussian kernel
    prob = torch.exp(-0.5 * ((classes - scaled_mos) / scaled_std) ** 2)

    # 行归一化，保证每行和为 1 / row-normalize so each row sums to 1
    return prob / prob.sum(dim=1, keepdim=True)


def expected_score(
    prob: torch.Tensor,
    num_buckets: int = 10,
    max_score: float = 100.0,
) -> torch.Tensor:
    """把概率分布还原为连续分数（SRCC/PLCC 用）/ Convert PDF back to scalar score.

    Args:
        prob:        [B, num_buckets] 概率分布（softmax 后）/ softmax distribution
        num_buckets: 桶数，必须与训练时一致 / must match training
        max_score:   MOS 上界 / upper bound of MOS

    Returns:
        [B] 连续分数 / scalar score
    """
    classes = torch.arange(
        1, num_buckets + 1, device=prob.device, dtype=torch.float32
    )
    # 期望桶 = Σ p_i * i / expected bin = Σ p_i * i
    expected_bucket = (prob * classes).sum(dim=1)
    # 映射回 [0, max_score] / map back to MOS scale
    return (expected_bucket / num_buckets) * max_score


def sanity_check_distribution(prob: torch.Tensor, tol: float = 1e-5) -> None:
    """断言每行概率和约为 1 / Assert each row sums to ~1.

    训练前应调用一次，发现 softmax 遗漏等低级错误。
    Run once before training to catch missing softmax or similar bugs.
    """
    row_sum = prob.sum(dim=1)
    if not torch.allclose(row_sum, torch.ones_like(row_sum), atol=tol):
        max_err = (row_sum - 1.0).abs().max().item()
        raise ValueError(
            f"分布和不为 1 / distribution does not sum to 1. max |sum-1|={max_err:.3e}"
        )

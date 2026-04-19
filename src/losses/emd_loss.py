"""
Squared Earth Mover's Distance (EMD²) 损失函数 / Squared EMD loss.

参考 / Reference:
    Hou et al. "Squared Earth Mover's Distance-based Loss for Training
    Deep Neural Networks." arXiv 2016.

为什么不用 MSE？/ Why not MSE?
    IQA 评分是序数变量，把分数 9 误判成 10 应该比误判成 2 代价小。
    MSE 对类别概率只看差值，不看顺序；EMD 对 CDF 的差做惩罚，
    自然地对"邻近桶"的错误更宽容。

    IQA scores are ordinal — confusing 9 with 10 should cost less than
    confusing 9 with 2. MSE on class probabilities is order-blind;
    EMD operates on CDFs, so near-neighbor mistakes are cheap while
    far-off ones are expensive. This matches human perception.

数值稳定性 / Numerical stability:
    - cumsum 在 fp16 下易下溢，混合精度必须用 bfloat16
    - cumsum underflows in fp16 — AMP must use bfloat16
    - 在 pow(1/r) 前加 1e-8，避免差为 0 时梯度 NaN
    - add 1e-8 before pow(1/r) to prevent NaN when difference==0
    - 输入前必须 softmax / input MUST be softmax-normalized

参见技术文档 Section 5 / See technical spec Section 5.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EMDLoss(nn.Module):
    """Squared Earth Mover's Distance for ordinal distributions.

    Args:
        r:       距离阶数，r=2 为平方 EMD（论文推荐）/ order, r=2 = squared EMD
        eps:     数值稳定项 / numerical safety term
    """

    def __init__(self, r: int = 2, eps: float = 1e-8) -> None:
        super().__init__()
        self.r = r
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """计算 EMD 损失 / Compute EMD loss.

        Args:
            pred:   [B, num_buckets] 模型输出的 softmax 分布，每行和为 1
                    model output after softmax, each row sums to 1
            target: [B, num_buckets] Gaussian 目标分布，每行和为 1
                    Gaussian target distribution, each row sums to 1

        Returns:
            标量损失（batch 平均）/ scalar loss averaged over batch
        """
        # 计算累积分布函数 CDF / compute CDFs
        # 注：cumsum 在 bf16 下数值稳定，在 fp16 下会下溢
        # Note: cumsum is safe in bf16 but underflows in fp16
        cdf_pred = torch.cumsum(pred, dim=1)
        cdf_target = torch.cumsum(target, dim=1)

        # |Δ CDF|^r，在桶维度上取平均 / |ΔCDF|^r averaged over buckets
        emd = torch.pow(
            torch.abs(cdf_pred - cdf_target), self.r
        ).mean(dim=1)

        # 加 eps 再开 1/r 次方，防止差为 0 时梯度 NaN
        # Add eps before pow(1/r) to avoid NaN gradient at zero-difference
        return torch.pow(emd + self.eps, 1.0 / self.r).mean()

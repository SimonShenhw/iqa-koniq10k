"""
Grad-CAM 可视化（针对 CNN / ResNet-50）/ Grad-CAM for CNN / ResNet-50.

基于 pytorch-grad-cam 库，对指定桶的预测计算梯度加权特征图。
Uses pytorch-grad-cam to compute gradient-weighted feature maps for a
target bucket prediction.

对 IQA 的含义 / Semantics for IQA:
    "模型认为图像的哪些区域最影响它对这个质量分数的判断？"
    分布模式下，target = argmax(pred_prob) 对应的桶索引
    Distribution mode: target = the bucket index with highest predicted prob.

参见技术文档 Section 9.1 / See spec Section 9.1.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError as e:
    raise ImportError(
        "pytorch-grad-cam 未安装 / not installed. "
        "Run: pip install pytorch-grad-cam"
    ) from e


class ResNetGradCAM:
    """ResNet-50 Grad-CAM 热力图生成器 / ResNet-50 Grad-CAM generator.

    Args:
        model:          待解释的模型，应提供 gradcam_target_layer() 方法
                        model under test, must provide gradcam_target_layer()
        target_layer:   可选手动指定目标层 / optional manual target layer override
        device:         推理设备 / device
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        device: str = "cuda",
    ):
        self.model = model.eval().to(device)
        self.device = device

        # 优先用模型暴露的接口 / prefer model's own interface
        if target_layer is None:
            if not hasattr(model, "gradcam_target_layer"):
                raise AttributeError(
                    "模型未实现 gradcam_target_layer() — "
                    "请手动传入 target_layer / manually pass target_layer"
                )
            target_layer = model.gradcam_target_layer()

        self.cam = GradCAM(model=self.model, target_layers=[target_layer])

    @torch.enable_grad()
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_bucket: Optional[Union[int, torch.Tensor]] = None,
    ) -> np.ndarray:
        """生成 Grad-CAM 热力图 / produce Grad-CAM heatmap.

        Args:
            input_tensor: [1, 3, H, W] 已归一化的输入 / normalized input
            target_bucket: 目标桶索引；None 时自动取预测最大桶
                           target bucket id; None → use argmax of prediction

        Returns:
            [H, W] 灰度热力图 ∈ [0, 1] / grayscale heatmap in [0, 1]
        """
        assert input_tensor.dim() == 4 and input_tensor.size(0) == 1, (
            "当前实现仅支持单图 / single-image batch only"
        )
        input_tensor = input_tensor.to(self.device)

        # 自动决定目标桶 / auto-pick target bucket
        if target_bucket is None:
            with torch.no_grad():
                prob = self.model(input_tensor)
                target_bucket = int(prob.argmax(dim=-1).item())
        elif isinstance(target_bucket, torch.Tensor):
            target_bucket = int(target_bucket.item())

        targets = [ClassifierOutputTarget(target_bucket)]
        grayscale = self.cam(input_tensor=input_tensor, targets=targets)
        # 形状 [1, H, W]，取第一个样本 / shape [1, H, W], take first sample
        return grayscale[0]

    @staticmethod
    def overlay(
        cam: np.ndarray,
        rgb_image: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """把热力图叠加到原图上 / overlay CAM onto RGB image.

        Args:
            cam:       [H, W] Grad-CAM 灰度图 / heatmap
            rgb_image: [H, W, 3] 归一到 [0,1] 的 RGB 图 / RGB image in [0,1]
            alpha:     热力图透明度 / heatmap opacity

        Returns:
            [H, W, 3] uint8 叠加图 / uint8 overlay
        """
        return show_cam_on_image(rgb_image, cam, use_rgb=True, image_weight=1 - alpha)

"""
PyTorch → ONNX 导出 / PyTorch → ONNX export.

用法 / Usage:
    python -m src.deploy.export_onnx \
        --config configs/resnet50_emd.yaml \
        --ckpt outputs/checkpoints/resnet50_best.pt \
        --output resnet50_iqa.onnx

对 ViT，会自动调用 onnxruntime-tools 的 Transformer 图优化器：
    - 融合 LayerNorm / GELU / Attention / SkipLayerNorm
    - 生成 *_optimized.onnx 与 *_optimized_fp16.onnx

For ViT, automatically invokes onnxruntime-tools Transformer optimizer:
    - Fuses LayerNorm / GELU / Attention / SkipLayerNorm
    - Emits *_optimized.onnx and *_optimized_fp16.onnx

参见技术文档 Section 10.1 与 10.1b / See spec Sections 10.1 and 10.1b.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows 控制台默认 GBK，PyTorch 2.11 ONNX exporter 会打印 ✅ 等 Unicode，
# 强制 stdout 为 utf-8 以避免 UnicodeEncodeError。
# Force UTF-8 stdout — PyTorch 2.11 ONNX exporter prints Unicode (e.g. ✅)
# that would crash Windows' default GBK console.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import yaml

from ..models import build_model


def load_config(path: str) -> dict:
    """读取 YAML 配置 / load YAML config."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_size: int = 224,
    device: str = "cuda",
    opset: int = 17,
) -> None:
    """基础 ONNX 导出（无图优化）/ Basic ONNX export with no graph optimization.

    Args:
        model:       已加载权重、已 .eval() 的模型 / model with weights, in eval mode
        output_path: 输出 .onnx 路径 / output .onnx path
        input_size:  输入正方形边长 / square input edge
        device:      cpu / cuda
        opset:       ONNX opset 版本 / ONNX opset version
    """
    model = model.eval().to(device)
    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)

    # 动态 batch 维度 → 部署时可以跑 batch>1
    # Dynamic batch axis → can run batch>1 at deploy time
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["prob"],
        dynamic_axes={"input": {0: "batch"}, "prob": {0: "batch"}},
        do_constant_folding=True,
    )
    print(f"[OK] 已导出 / exported: {output_path}")

    # 校验 ONNX 模型 / validate ONNX model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"[OK] ONNX 校验通过 / ONNX checker passed")
    except ImportError:
        print("[注意] 未安装 onnx，跳过校验 / onnx not installed, skip checking")


def apply_transformer_optimizer(
    onnx_path: str,
    num_heads: int = 12,
    hidden_size: int = 768,
    to_fp16: bool = True,
) -> None:
    """对 ViT 导出图应用 Transformer 图优化 / Apply transformer graph optimization for ViT.

    作用 / Effect:
        把每个注意力 block 的 40+ 原生 op（Reshape/Transpose/MatMul/Softmax）
        融合成 Attention / LayerNorm / GELU / SkipLayerNorm 等融合核。
        预期延迟下降 30-50% + FP16 再下降一半。

        Fuses 40+ primitive ops per attention block into Attention, LayerNorm,
        GELU, SkipLayerNorm kernels. Expected 30-50% latency reduction +
        another ~50% from FP16.
    """
    try:
        from onnxruntime.transformers import optimizer
        from onnxruntime.transformers.fusion_options import FusionOptions
    except ImportError:
        print(
            "[警告 / WARN] onnxruntime-tools 未安装，跳过 Transformer 优化\n"
            "  Run: pip install onnxruntime-tools"
        )
        return

    fusion_options = FusionOptions("vit")
    fusion_options.enable_gelu = True
    fusion_options.enable_layer_norm = True
    fusion_options.enable_attention = True
    fusion_options.enable_skip_layer_norm = True

    # 最大优化等级 / max opt level
    optimized_model = optimizer.optimize_model(
        onnx_path,
        model_type="vit",
        num_heads=num_heads,
        hidden_size=hidden_size,
        optimization_options=fusion_options,
        opt_level=99,
    )

    opt_path = onnx_path.replace(".onnx", "_optimized.onnx")
    optimized_model.save_model_to_file(opt_path)
    print(f"[OK] Transformer 优化已保存 / saved optimized model: {opt_path}")

    if to_fp16:
        # 原地转 FP16 / in-place FP16 conversion
        optimized_model.convert_float_to_float16()
        fp16_path = onnx_path.replace(".onnx", "_optimized_fp16.onnx")
        optimized_model.save_model_to_file(fp16_path)
        print(f"[OK] FP16 版本已保存 / saved FP16 model: {fp16_path}")


def convert_resnet_fp16(onnx_path: str) -> None:
    """ResNet-50 简单 FP16 转换（不做图融合）/ Simple FP16 for ResNet-50 (no fusion).

    适用于 ResNet-50 等无 Transformer 结构的模型。
    For ResNet-50 and other non-transformer models.
    """
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError:
        print("[警告] onnxconverter_common 未安装 / not installed — 跳过")
        return

    model_fp32 = onnx.load(onnx_path)
    # keep_io_types=True：输入输出仍为 FP32，内部计算转 FP16
    # keep_io_types=True: IO stays FP32, internals convert to FP16
    model_fp16 = float16.convert_float_to_float16(model_fp32, keep_io_types=True)

    fp16_path = onnx_path.replace(".onnx", "_fp16.onnx")
    onnx.save(model_fp16, fp16_path)
    print(f"[OK] FP16 版本已保存 / saved FP16 model: {fp16_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX 导出 / ONNX export")
    parser.add_argument("--config", required=True, help="YAML 配置 / YAML config")
    parser.add_argument("--ckpt", required=True, help="checkpoint .pt 路径 / ckpt path")
    parser.add_argument(
        "--output", default=None,
        help="输出 .onnx 路径 / output .onnx path (默认 outputs/<exp>.onnx)"
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"],
        help="导出所用设备 / export device"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    model_name = model_cfg["name"]

    # 构建模型并加载权重 / build model and load weights
    model = build_model(**model_cfg)
    # weights_only=False：ckpt 含 numpy 标量等，我们自己保存的可信
    # weights_only=False: our own ckpt contains numpy scalars, trusted
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # 兼容直接 state_dict 或 checkpoint dict / accept either format
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    print(f"[OK] 已加载权重 / loaded checkpoint: {args.ckpt}")

    # 输出路径 / output path
    if args.output is None:
        exp_name = cfg["experiment"]["name"]
        out_dir = Path(cfg["experiment"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(out_dir / f"{exp_name}.onnx")

    # 基础导出 / basic export
    input_size = cfg["data"].get("image_size", 224)
    export_to_onnx(model, args.output, input_size=input_size, device=args.device)

    # 按模型类型选择优化路径 / optimize based on model type
    if model_name == "vit_lora":
        # ViT：Transformer 图融合 + FP16
        apply_transformer_optimizer(
            args.output,
            num_heads=12,
            hidden_size=768,
            to_fp16=True,
        )
    elif model_name in ("resnet50_emd", "baseline_cnn"):
        # 非 Transformer 模型：直接 FP16 转换
        convert_resnet_fp16(args.output)


if __name__ == "__main__":
    main()

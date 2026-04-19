"""
多后端延迟基准测试 / Multi-backend latency benchmark.

覆盖 / Covers:
    - PyTorch eager (FP32 / BF16)
    - PyTorch torch.compile (BF16)
    - ONNX Runtime + CUDA EP (FP32 / FP16)
    - ONNX Runtime + DirectML EP (Windows 原生 GPU / Windows-native GPU)
    - TensorRT (FP16 / INT8) —— 可选（桌面 5090 上跑）

    NOTE: 技术文档 Section 10.0 指出 Blackwell 笔记本的 TensorRT 可能比
    ORT CUDA 慢 10-15 倍。本基准优先报告 ORT 数字，TensorRT 仅在桌面上启用。

    Spec Section 10.0 notes TensorRT on Blackwell laptops may be 10-15x
    slower than ORT CUDA. We report ORT numbers first; TensorRT is optional
    and desktop-only.

用法 / Usage:
    python -m src.deploy.tensorrt_benchmark \
        --onnx outputs/resnet50_emd.onnx \
        --backends ort-cuda,ort-dml,ort-cuda-fp16 \
        --warmup 100 --iters 1000

参见技术文档 Section 10.2 ~ 10.6 / See spec Sections 10.2 - 10.6.
"""

from __future__ import annotations

import argparse
import time
from typing import Callable, Dict, List, Tuple

import numpy as np


# ======================================================================
# 计时工具 / Timing helpers
# ======================================================================
def benchmark_callable(
    fn: Callable[[], None],
    warmup: int = 100,
    iters: int = 1000,
) -> Tuple[float, float]:
    """对给定推理闭包计时 / Time a forward closure.

    Args:
        fn:     一次前向推理的无参数闭包 / zero-arg closure doing one forward
        warmup: 预热次数 / warmup iterations
        iters:  正式测量次数 / measured iterations

    Returns:
        (mean_ms, std_ms) 单次推理延迟的均值 ± 标准差
    """
    # 预热（编译、缓存、cudnn autotune）/ warmup (compile, cache, cudnn autotune)
    for _ in range(warmup):
        fn()

    # 逐次计时，存下后算均值/方差 / measure per-iter then aggregate
    timings_ms: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(timings_ms)
    return float(arr.mean()), float(arr.std())


# ======================================================================
# 后端 / Backends
# ======================================================================
def bench_ort(
    onnx_path: str,
    provider: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    provider_options: dict = None,
    warmup: int = 100,
    iters: int = 1000,
) -> Dict[str, float]:
    """ONNX Runtime 推理基准 / ONNX Runtime inference benchmark.

    Args:
        onnx_path:        ONNX 文件路径 / path to .onnx
        provider:         'CUDAExecutionProvider' / 'DmlExecutionProvider' / 'CPUExecutionProvider'
        input_shape:      输入形状 / input shape
        provider_options: provider 额外参数 / extra provider options
        warmup / iters:   同 benchmark_callable
    """
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Provider 组装：优先使用目标 provider，CPU 兜底
    # Provider list: target first, CPU as fallback
    if provider_options:
        providers = [(provider, provider_options), "CPUExecutionProvider"]
    else:
        providers = [provider, "CPUExecutionProvider"]

    sess = ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=providers,
    )

    # 输入名与真实输入 / input name and dummy input
    input_name = sess.get_inputs()[0].name

    # 检查模型输入类型（FP32 或 FP16）/ check input dtype
    input_type = sess.get_inputs()[0].type  # e.g. 'tensor(float)' or 'tensor(float16)'
    np_dtype = np.float16 if "float16" in input_type else np.float32
    dummy = np.random.randn(*input_shape).astype(np_dtype)

    def _run():
        sess.run(None, {input_name: dummy})

    mean_ms, std_ms = benchmark_callable(_run, warmup=warmup, iters=iters)
    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "throughput": 1000.0 / mean_ms,   # img/s at batch=1
    }


def bench_pytorch_eager(
    model,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    dtype: str = "fp32",
    device: str = "cuda",
    warmup: int = 100,
    iters: int = 1000,
) -> Dict[str, float]:
    """PyTorch eager 模式基准 / PyTorch eager benchmark.

    Args:
        model: 已 eval 的 nn.Module / nn.Module in eval mode
        dtype: 'fp32' 或 'bf16'
    """
    import torch

    model = model.eval().to(device)
    torch_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype]

    dummy = torch.randn(*input_shape, device=device, dtype=torch_dtype)
    if dtype == "bf16":
        model = model.to(torch_dtype)

    @torch.no_grad()
    def _run():
        _ = model(dummy)
        torch.cuda.synchronize()

    mean_ms, std_ms = benchmark_callable(_run, warmup=warmup, iters=iters)
    return {"mean_ms": mean_ms, "std_ms": std_ms, "throughput": 1000.0 / mean_ms}


def bench_pytorch_compile(
    model,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = "cuda",
    warmup: int = 100,
    iters: int = 1000,
) -> Dict[str, float]:
    """torch.compile + BF16 基准 / torch.compile + BF16 benchmark.

    注意 / Note:
        torch.compile 的首次调用会很慢（编译），所以 warmup 必须充足。
        The first call triggers compilation — make sure warmup is generous.
    """
    import torch

    model = model.eval().to(device).to(torch.bfloat16)
    compiled = torch.compile(model, mode="default")

    dummy = torch.randn(*input_shape, device=device, dtype=torch.bfloat16)

    @torch.no_grad()
    def _run():
        _ = compiled(dummy)
        torch.cuda.synchronize()

    mean_ms, std_ms = benchmark_callable(_run, warmup=max(warmup, 200), iters=iters)
    return {"mean_ms": mean_ms, "std_ms": std_ms, "throughput": 1000.0 / mean_ms}


def bench_tensorrt(
    onnx_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    fp16: bool = True,
    warmup: int = 100,
    iters: int = 1000,
) -> Dict[str, float]:
    """TensorRT 引擎基准（可选，需 tensorrt Python 包）.

    TensorRT engine benchmark (optional, requires 'tensorrt' Python package).

    NOTE: Blackwell 笔记本已知存在性能退化，见技术文档 Section 10.0。
          Known perf regression on Blackwell laptops; see spec Section 10.0.
    """
    try:
        import tensorrt as trt                     # noqa: F401 — 运行时可用性检查
        import pycuda.driver as cuda               # noqa: F401
        import pycuda.autoinit                     # noqa: F401
    except ImportError:
        return {"error": "tensorrt / pycuda 未安装 — 跳过 / not installed, skipped"}

    # TensorRT 引擎构建与推理代码较复杂，建议用 trtexec 生成 .engine 后
    # 再用 tensorrt.Runtime 加载。这里给出最小骨架；如需完整实现，
    # 参考 NVIDIA samples/python/classification。
    #
    # Building and running a TRT engine is lengthy — prefer generating a
    # .engine with trtexec and loading via tensorrt.Runtime. This function
    # returns a stub; a full impl lives in NVIDIA's samples/python/classification.
    return {
        "error": (
            "需手动用 trtexec 生成 .engine 后再实现 host-device 拷贝循环 "
            "Implement host-device copy loop after `trtexec` .engine generation."
        ),
    }


# ======================================================================
# 主入口 / Main
# ======================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="多后端延迟基准 / multi-backend bench")
    parser.add_argument("--onnx", required=True, help="ONNX 模型路径 / ONNX model path")
    parser.add_argument(
        "--backends",
        default="ort-cuda,ort-dml",
        help="逗号分隔的后端列表 / comma-separated: "
             "ort-cuda, ort-cuda-fp16, ort-dml, ort-cpu, tensorrt-fp16"
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    args = parser.parse_args()

    input_shape = (args.batch, 3, args.size, args.size)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    results: Dict[str, Dict[str, float]] = {}
    for bk in backends:
        print(f"\n=== 基准后端 / Benchmarking: {bk} ===")
        try:
            if bk == "ort-cuda":
                r = bench_ort(
                    args.onnx,
                    provider="CUDAExecutionProvider",
                    provider_options={"cudnn_conv_algo_search": "EXHAUSTIVE"},
                    input_shape=input_shape,
                    warmup=args.warmup, iters=args.iters,
                )
            elif bk == "ort-cuda-fp16":
                # 把 _fp16.onnx 传进来 / pass the _fp16.onnx variant
                fp16_path = args.onnx.replace(".onnx", "_fp16.onnx")
                r = bench_ort(
                    fp16_path,
                    provider="CUDAExecutionProvider",
                    input_shape=input_shape,
                    warmup=args.warmup, iters=args.iters,
                )
            elif bk == "ort-dml":
                # Windows 原生 DirectML，无需 CUDA / Windows-native, no CUDA
                r = bench_ort(
                    args.onnx,
                    provider="DmlExecutionProvider",
                    input_shape=input_shape,
                    warmup=args.warmup, iters=args.iters,
                )
            elif bk == "ort-cpu":
                r = bench_ort(
                    args.onnx,
                    provider="CPUExecutionProvider",
                    input_shape=input_shape,
                    warmup=args.warmup, iters=args.iters,
                )
            elif bk == "tensorrt-fp16":
                r = bench_tensorrt(
                    args.onnx,
                    input_shape=input_shape,
                    fp16=True,
                    warmup=args.warmup, iters=args.iters,
                )
            else:
                r = {"error": f"未知后端 / unknown backend: {bk}"}
        except Exception as e:  # 单个后端失败不影响其他 / don't let one crash the rest
            r = {"error": str(e)}

        results[bk] = r
        print(r)

    # 汇总表 / summary table
    print("\n" + "=" * 72)
    print(f"{'Backend':<22} {'Mean (ms)':>12} {'Std (ms)':>12} {'Throughput (img/s)':>22}")
    print("-" * 72)
    for bk, r in results.items():
        if "error" in r:
            print(f"{bk:<22} [error] {r['error']}")
        else:
            print(
                f"{bk:<22} {r['mean_ms']:>12.3f} {r['std_ms']:>12.3f} "
                f"{r['throughput']:>22.2f}"
            )
    print("=" * 72)


if __name__ == "__main__":
    main()

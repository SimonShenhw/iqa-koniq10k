"""
Day 1 环境验证脚本 / Day-1 environment verification script.

作用 / Purpose:
- 确认 PyTorch 版本、CUDA 可用性、GPU 型号与 Compute Capability
- Verify PyTorch version, CUDA availability, GPU name & compute capability

- 针对 RTX 50 系列必须返回 sm_120 (即 (12, 0))
- For RTX 50-series, compute capability MUST be (12, 0)

- 对 bfloat16 矩阵乘做一次烟雾测试，确认 sm_120 内核存在
- Run a bfloat16 matmul smoke test to confirm sm_120 kernels are present

用法 / Usage:
    python verify_env.py
"""

import sys

import torch


def main() -> None:
    # 基本信息打印 / Print basic info
    print("=" * 60)
    print("IQA 项目环境验证 / IQA project env verification")
    print("=" * 60)
    print(f"Python:           {sys.version.split()[0]}")
    print(f"PyTorch:          {torch.__version__}")
    print(f"CUDA available:   {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        # 致命错误：GPU 不可用 / Fatal: GPU not available
        print("[错误 / ERROR] CUDA 不可用 — 无法继续 / CUDA unavailable, abort.")
        sys.exit(1)

    print(f"CUDA version:     {torch.version.cuda}")
    print(f"Device:           {torch.cuda.get_device_name(0)}")

    # Compute capability 检查 / CC check
    cc = torch.cuda.get_device_capability(0)
    print(f"Compute capability: {cc}")

    # RTX 50 系列应返回 (12, 0) / Expect (12, 0) on RTX 50-series
    if cc == (12, 0):
        print("[OK] 检测到 Blackwell sm_120 — 符合预期 / Blackwell sm_120 detected.")
    elif cc >= (8, 0):
        print(f"[注意 / NOTE] 非 Blackwell GPU (cc={cc})，项目可运行但非目标硬件")
        print(f"              Non-Blackwell GPU — runnable, but not the target.")
    else:
        print(f"[警告 / WARN] Compute capability {cc} 过低，可能不支持 bfloat16")
        print(f"              Low compute capability — may not support bfloat16.")

    # bfloat16 matmul 烟雾测试 / bfloat16 matmul smoke test
    # 这一步验证 sm_120 内核真正存在 / This verifies sm_120 kernels are actually present
    print("-" * 60)
    print("运行 bfloat16 矩阵乘烟雾测试 / Running bfloat16 matmul smoke test...")
    try:
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
        y = x @ x.T
        torch.cuda.synchronize()
        print(f"[OK] bfloat16 matmul 成功 / succeeded — shape: {tuple(y.shape)}")
    except Exception as e:
        # 常见失败原因：PyTorch 版本过低未包含 sm_120 内核
        # Common failure: PyTorch version too old, sm_120 kernels missing
        print(f"[错误 / ERROR] bfloat16 matmul 失败 / failed: {e}")
        print("  → 检查 PyTorch 是否为 2.9.1+ 且使用 cu130 构建")
        print("  → Check PyTorch is 2.9.1+ and built with cu130")
        sys.exit(1)

    # 显存信息 / VRAM info
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"显存总量 / Total VRAM: {total_gb:.2f} GB")

    # 检查 TF32 开关 / Check TF32 flags
    print("-" * 60)
    print("Tensor Core 配置 / Tensor Core config:")
    print(f"  cudnn.allow_tf32           : {torch.backends.cudnn.allow_tf32}")
    print(f"  cuda.matmul.allow_tf32     : {torch.backends.cuda.matmul.allow_tf32}")
    print("  → 训练脚本会启用 'high' 精度以激活 TF32 / train.py enables TF32 via 'high'.")

    print("=" * 60)
    print("环境验证通过 / Environment OK. 可以开始训练 / ready to train.")
    print("=" * 60)


if __name__ == "__main__":
    main()

# IQA Project — Technical Specification

**Project**: Deep Learning for Image Quality Assessment on KonIQ-10k
**Course**: AAI 6640 Final Project
**Team**: Haowei Shen, Xuezhen Jin, Peter Adranly
**Timeline**: 8 weeks (Week 1 → Presentation)
**Last Updated**: 2026-04-18 (v4 — added TF32 Tensor Core activation + precision stack layering with bf16 AMP)

---

## 1. Project Goals

### Research Questions
1. **Distribution vs. Mean Prediction**: Can EMD loss on probability distributions outperform MSE regression on mean scores for in-the-wild IQA?
2. **XAI for Defect Visualization**: Can CNN Grad-CAM and ViT Attention Rollout qualitatively highlight visual drawbacks (blur, exposure) aligned with human intuition?
3. **Efficiency vs. Performance**: How to balance prediction quality with deployment latency via quantization and inference engine optimization?

### Deliverables
- Three trained models (Baseline CNN, ResNet-50+EMD, ViT-B/16+LoRA+EMD)
- Systematic comparison: in-dataset + cross-dataset SRCC/PLCC
- Qualitative heatmap visualizations (Grad-CAM vs Attention Rollout)
- Deployment benchmark: PyTorch FP32 → ONNX → TensorRT FP16/INT8 latency comparison
- APA-formatted report + presentation deck

---

## 2. Environment Setup

### Hardware
- **Training**: RTX 5090 desktop (32GB VRAM, 9950X3D CPU)
- **Dev/Test**: ROG laptop (RTX 5080 Laptop)
- **Storage**: Reserve ~80GB for KonIQ-10k + SPAQ + checkpoints

### Software Stack

**CRITICAL: RTX 50-series (Blackwell, sm_120) hardware compatibility**

The RTX 5090 and RTX 5080 Laptop are **Blackwell architecture with Compute Capability sm_120**. This requires PyTorch builds compiled with `TORCH_CUDA_ARCH_LIST` including sm_120. **PyTorch ≤ 2.5 with cu124 or earlier will NOT work** — you will get `UserWarning: NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation` and either crash or fall back to CPU.

**Minimum required stack (as of April 2026)**:
- PyTorch 2.9.1 + cu130 (stable) — **recommended default**
- PyTorch 2.10.0 + cu130 (stable, newer) — acceptable
- PyTorch 2.11 + cu130 (upcoming, will make CUDA 13 officially stable)
- Python 3.12+ (3.13 recommended for Blackwell)

**Install command (all three team members must use this exact command)**:
```bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu130
```

> Note: cu128 also works but cu130 is more forward-compatible. Do NOT mix cu128 and cu130 environments across team members — it creates subtle reproducibility bugs in `torch.compile` graphs.

**Full dependency list**:
```
Python 3.12+ (3.13 preferred)
torch==2.9.1           # With cu130
torchvision==0.24.1
torchaudio==2.9.1
timm >= 1.0            # ViT models
peft >= 0.11           # LoRA
pytorch-grad-cam       # CNN heatmaps
onnx >= 1.17
onnxruntime-gpu >= 1.20
onnxruntime-directml   # Windows-native GPU backend (fallback / comparison)
# TensorRT: optional — install only if TensorRT-RTX is available for your setup
pandas, numpy, scipy
scikit-learn           # SRCC/PLCC via scipy.stats
matplotlib, seaborn
wandb                  # Experiment tracking (or tensorboard)
pillow
tqdm
pyyaml                 # Config files
```

**Environment verification script (run on Day 1 for each machine)**:
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
# Expected for RTX 5090: (12, 0)
# Expected for RTX 5080 Laptop: (12, 0)

# Smoke test: ensure sm_120 kernels exist
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
y = x @ x.T
print(f"bfloat16 matmul OK, result shape: {y.shape}")
```

### Repo Structure
```
iqa-project/
├── configs/
│   ├── baseline_cnn.yaml
│   ├── resnet50_emd.yaml
│   └── vit_lora_emd.yaml
├── data/
│   ├── koniq10k/           # Images + metadata
│   └── spaq/               # Cross-dataset evaluation
├── src/
│   ├── datasets/
│   │   ├── koniq10k.py
│   │   └── spaq.py
│   ├── models/
│   │   ├── baseline_cnn.py
│   │   ├── resnet50_emd.py
│   │   └── vit_lora.py
│   ├── losses/
│   │   └── emd_loss.py
│   ├── utils/
│   │   ├── distribution.py  # MOS → Gaussian target
│   │   ├── metrics.py       # SRCC, PLCC
│   │   └── multi_crop.py
│   ├── xai/
│   │   ├── gradcam.py
│   │   └── attention_rollout.py
│   ├── deploy/
│   │   ├── export_onnx.py
│   │   └── tensorrt_benchmark.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/              # EDA + visualization
├── outputs/                # Checkpoints, logs, heatmaps
├── requirements.txt
├── README.md
└── IQA_Technical_Spec.md   # This document
```

---

## 3. Dataset Specification

### KonIQ-10k (Primary)
- **Source (official)**: https://database.mmsp-kn.de/koniq-10k-database.html
- **Mirror (backup)**: Kaggle — search `KonIQ-10k` on kaggle.com and use Kaggle CLI:
  ```bash
  # The official server (Konstanz University, Germany) frequently throttles
  # or times out for non-EU users. Use Kaggle as the primary download path.
  pip install kaggle
  # Put kaggle.json in ~/.kaggle/ (or %USERPROFILE%\.kaggle\ on Windows)
  kaggle datasets download -d <kaggle-dataset-slug>/koniq10k
  unzip koniq10k.zip -d data/koniq10k/
  ```
- **Size**: 10,073 images, 1024×768 resolution, ~15-25 GB depending on mirror
- **Labels**: MOS (1-100 scale), std, per-attribute scores (brightness, colorfulness, contrast, sharpness)
- **Split**: 8:1:1 by `image_id` (deterministic seed: 42). **Do NOT use random split on rows — risks leakage if duplicates exist.**
- **Download checklist (Day 1)**: Try official first (5 min timeout), fall back to Kaggle if throttled. Verify file count (10,073) and checksums after download.

### SPAQ (Cross-Dataset Evaluation)
- **Source**: https://github.com/h4nwei/SPAQ
- **Size**: ~11K smartphone images
- **Use**: Test-only, zero-shot inference with all three trained models

### Data Pipeline Rules
- **Resize**: 512×384 for training (preserves aspect ratio closer to original 1024×768)
- **Train crop**: `RandomResizedCrop(224, scale=(0.5, 1.0))` — DO NOT use default `(0.08, 1.0)`, too aggressive for IQA
- **Augmentations allowed**: `HorizontalFlip` only
- **Augmentations FORBIDDEN**: `GaussianBlur`, `ColorJitter`, `RandomErasing`, any JPEG compression — these corrupt the quality label
- **Normalization**: ImageNet stats `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
- **Test-time augmentation (TTA)**: 5-crop (4 corners + center) OR 25-crop (5×5 grid), average probability distributions, then compute expected score

### DataLoader Configuration (CRITICAL for 9950X3D + RTX 5090)

With RTX 5090's memory bandwidth, JPEG I/O becomes the bottleneck on 10K images. The default `num_workers=0` or `num_workers=4` will underfeed the GPU and waste 80%+ of the hardware's throughput. Use the following config:

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset,
    batch_size=128,           # ResNet-50 setting; adjust per model (Section 7)
    shuffle=True,
    num_workers=16,           # 9950X3D has 32 threads; 16 workers is safe + fast
    pin_memory=True,          # Page-locked memory for faster CPU→GPU transfer
    persistent_workers=True,  # Avoid worker respawn overhead each epoch
    prefetch_factor=4,        # Each worker prefetches 4 batches ahead
    drop_last=True,           # Stable batch shape for torch.compile
)

val_loader = DataLoader(
    dataset_val,
    batch_size=256,           # Larger batch OK at eval (no backward pass)
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)
```

**On the ROG laptop (RTX 5080 Laptop)**: Reduce `num_workers` to 8 (laptop CPUs have fewer cores + thermal throttling under sustained load).

**Windows-specific gotcha**: `persistent_workers=True` + `num_workers>0` sometimes causes `RuntimeError: DataLoader worker (pid X) is killed by signal` on Windows. If you hit this, the fix is: set `num_workers=0` for debugging first, then ramp up. Also ensure all DataLoader code is inside `if __name__ == "__main__":` guard.

### Windows Multi-Processing Memory Leak (9950X3D + Windows specific)

This is the single most insidious bug that will hit this project if not prevented upfront. On Windows, PyTorch defaults to `spawn` start method (vs `fork` on Linux). `spawn` + `persistent_workers=True` + `pin_memory=True` has a known issue where each DataLoader worker accumulates memory (both RAM and pinned host memory) across epochs because the workers never release their page-locked buffers properly. Symptom: training works fine for 3-5 epochs, then OOM around epoch 8-12 with no obvious cause.

**Mandatory mitigation** — the first thing `train.py` and `evaluate.py` must do:

```python
import torch
import multiprocessing as mp
import random
import numpy as np

def seed_worker(worker_id: int):
    """Ensure each DataLoader worker has a distinct but reproducible seed.
    Without this, all workers generate the same RandomResizedCrop output,
    which silently corrupts data augmentation."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    # MUST be the first thing before any CUDA or DataLoader code
    mp.set_start_method("spawn", force=True)

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = False  # Keep False for speed; True for strict repro
    torch.backends.cudnn.benchmark = True       # Auto-tune conv algos for fixed shapes

    # Unleash Blackwell Tensor Cores for FP32 matmul
    # By default, PyTorch uses strict IEEE FP32 for matmul — this does NOT use Tensor Cores
    # and leaves ~60-70% of peak compute unused on RTX 50-series.
    # Setting 'high' routes FP32 matmul through TF32 (10-bit mantissa) via Tensor Cores.
    #
    # Naming is counter-intuitive:
    #   'highest' = pure FP32, NO Tensor Core acceleration (default, slow)
    #   'high'    = TF32 via Tensor Core, ~0.001 relative error (what we want)
    #   'medium'  = bfloat16 matmul, more aggressive, use only for extreme speed
    torch.set_float32_matmul_precision("high")

    # Also enable TF32 for convolutions (cuDNN path — independent of the matmul setting above)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True  # Legacy API, still respected for safety

    # ... rest of training script
```

Then pass `worker_init_fn=seed_worker` to every DataLoader:
```python
g = torch.Generator()
g.manual_seed(42)

train_loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=16,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    drop_last=True,
    worker_init_fn=seed_worker,
    generator=g,
)
```

**Additional defense**: If OOM still occurs after 20+ epochs despite the above, reduce `num_workers` to 8 and remove `persistent_workers` as a last resort. The 30% I/O throughput loss is better than crashing at epoch 25.

---

## 4. MOS → Probability Distribution Conversion

### Key Function (`src/utils/distribution.py`)
```python
import torch

def generate_gaussian_target(
    mos: torch.Tensor,           # shape [B], raw MOS in [0, 100]
    std: torch.Tensor,           # shape [B], raw std in [0, 100]
    num_buckets: int = 10,
    max_score: float = 100.0,
    min_std: float = 0.5,        # prevents one-hot degeneration
) -> torch.Tensor:
    """Returns [B, num_buckets] probability distribution."""
    scaled_mos = (mos / max_score) * num_buckets
    scaled_std = torch.clamp(std / max_score * num_buckets, min=min_std)

    classes = torch.arange(
        1, num_buckets + 1, device=mos.device, dtype=torch.float32
    )

    scaled_mos = scaled_mos.unsqueeze(1)           # [B, 1]
    scaled_std = scaled_std.unsqueeze(1)           # [B, 1]

    prob = torch.exp(-0.5 * ((classes - scaled_mos) / scaled_std) ** 2)
    return prob / prob.sum(dim=1, keepdim=True)


def expected_score(
    prob: torch.Tensor,          # [B, num_buckets]
    num_buckets: int = 10,
    max_score: float = 100.0,
) -> torch.Tensor:
    """Convert distribution back to scalar score for SRCC/PLCC computation."""
    classes = torch.arange(
        1, num_buckets + 1, device=prob.device, dtype=torch.float32
    )
    expected_bucket = (prob * classes).sum(dim=1)
    return (expected_bucket / num_buckets) * max_score
```

### Decisions
- **Bucket count**: 10 (primary), ablate with 5 if time permits
- **min_std**: 0.5 bucket units — lower values collapse to near one-hot, losing EMD's purpose
- **Sanity check**: Assert `prob.sum(dim=1)` ≈ 1.0 before computing loss

---

## 5. EMD Loss Implementation

### Squared EMD (`src/losses/emd_loss.py`)
```python
import torch
import torch.nn as nn

class EMDLoss(nn.Module):
    """Squared Earth Mover's Distance for ordinal distributions.
    Reference: Hou et al., 'Squared Earth Mover's Distance-based Loss
    for Training Deep Neural Networks' (2016).
    """
    def __init__(self, r: int = 2):
        super().__init__()
        self.r = r  # r=2 is standard squared EMD

    def forward(
        self,
        pred: torch.Tensor,      # [B, num_buckets], must sum to 1 (post-softmax)
        target: torch.Tensor,    # [B, num_buckets], must sum to 1
    ) -> torch.Tensor:
        cdf_pred = torch.cumsum(pred, dim=1)
        cdf_target = torch.cumsum(target, dim=1)
        emd = torch.pow(
            torch.abs(cdf_pred - cdf_target), self.r
        ).mean(dim=1)
        return torch.pow(emd + 1e-8, 1.0 / self.r).mean()
```

### Numerical Stability Notes
- Use `bfloat16` NOT `float16` for mixed precision — `cumsum` in fp16 underflows easily
- Add `1e-8` before `pow(1/r)` to avoid NaN gradient when difference = 0
- Always softmax model output BEFORE passing to EMD loss

---

## 6. Model Specifications

### 6.1 Baseline CNN (`src/models/baseline_cnn.py`)
**Purpose**: Foundational foil for the comparison. Intentionally simple.

- **Architecture**: 4 conv blocks (Conv → BN → ReLU → MaxPool), then GAP + FC(1)
- **Channels**: 32 → 64 → 128 → 256
- **Input**: 224×224 RGB
- **Output**: Scalar MOS prediction
- **Loss**: MSE against normalized MOS (divide by 100 for numerical stability)
- **Parameters**: ~1-2M (reference point for efficiency comparison)

### 6.2 ResNet-50 + EMD (`src/models/resnet50_emd.py`)
**Purpose**: Primary "distribution prediction" model.

- **Backbone**: `torchvision.models.resnet50(weights="IMAGENET1K_V2")`
- **Head modification**: Replace `fc` with `Linear(2048, num_buckets)` + `Softmax(dim=-1)`
- **Freeze strategy**:
  - Phase 1 (epochs 0-4): Freeze all layers except head, LR=1e-3
  - Phase 2 (epochs 5-30): Unfreeze `layer3` + `layer4`, LR=1e-5 for backbone, 1e-4 for head
- **Loss**: EMD (squared)
- **Grad-CAM target layer**: `model.layer4[-1]` (last bottleneck block)

### 6.3 ViT-B/16 + LoRA + EMD (`src/models/vit_lora.py`)
**Purpose**: Test whether global self-attention beats local convolutions on in-the-wild IQA.

- **Base model**: `timm.create_model("vit_base_patch16_224", pretrained=True)`
- **Head modification**: Replace classifier with `Linear(768, num_buckets)` + `Softmax`
- **Pre-LoRA layer name verification** (MUST run before applying LoRA):
  ```python
  import timm
  vit_model = timm.create_model("vit_base_patch16_224", pretrained=True)

  # Print all modules that contain 'qkv' or 'attn' in their name
  # For vit_base_patch16_224, expect to see: blocks.{0..11}.attn.qkv
  for name, module in vit_model.named_modules():
      if 'qkv' in name or 'attn' in name:
          print(f"{name}: {type(module).__name__}")
  ```
  If you switch to a different backbone (e.g. `vit_large_patch14`, `eva02_base_patch14`, DeiT variants), the attention projection may be named differently (`q`, `k`, `v` as separate Linears, or `qkv_bias`, etc.). Always verify before LoRA config.

- **LoRA config**:
  ```python
  from peft import LoraConfig, get_peft_model
  config = LoraConfig(
      r=16,
      lora_alpha=32,
      target_modules=["qkv"],   # timm fuses Q/K/V into single Linear in ViT-B/16
      lora_dropout=0.1,
      bias="none",
      modules_to_save=["head"], # Train the new head fully
  )
  model = get_peft_model(vit_model, config)
  model.print_trainable_parameters()
  # Expected output: trainable params: ~850K || all params: ~86M || trainable%: ~0.99
  ```
- **Trainable parameter target**: <1% of base model (~800K params vs 86M)
- **Loss**: EMD (squared)
- **Attention Rollout**: Extract all 12 layer attention matrices, add identity, normalize, multiply sequentially, take CLS→patch row, reshape to 14×14, upsample bilinearly to 224×224

---

## 7. Training Configuration

### Hyperparameters
| Config | Baseline CNN | ResNet-50 | ViT-B/16 |
|--------|--------------|-----------|----------|
| Batch size | 256 | 128 | 64 |
| Epochs | 50 | 30 | 30 |
| Optimizer | AdamW | AdamW | AdamW |
| LR (head) | 1e-3 | 1e-4 | 1e-4 |
| LR (backbone) | N/A | 1e-5 | LoRA: 1e-4 |
| Weight decay | 1e-4 | 1e-4 | 0.05 |
| LR schedule | CosineAnnealing | CosineAnnealing w/ warmup | CosineAnnealing w/ warmup |
| Warmup epochs | 0 | 2 | 3 |
| Gradient clip | 1.0 | 1.0 | 1.0 |
| Precision | bfloat16 | bfloat16 | bfloat16 |
| `torch.compile` | Yes | Yes | Yes (may need disable if shape varies) |

### Training Loop Requirements
- Use `torch.cuda.amp.autocast(dtype=torch.bfloat16)` for mixed precision
- Use `torch.compile(model, mode="default")` — avoid `reduce-overhead` during training
- Save checkpoint every epoch + best-SRCC checkpoint separately
- Log to W&B: train loss, val SRCC, val PLCC, val MSE, LR, epoch time
- Early stopping: patience=5 epochs on val SRCC

### Precision Stack: Why TF32 AND bf16 AMP Together

These are NOT redundant — they cover different parts of the graph:

| Op type | Without TF32 | With TF32 only | With bf16 AMP only | **TF32 + bf16 AMP (recommended)** |
|---------|-------------|----------------|---------------------|----------------------------------|
| Conv / matmul inside autocast region | FP32 (slow) | TF32 (fast) | bf16 (fastest) | **bf16** |
| Conv / matmul outside autocast (e.g. some head ops, custom layers) | FP32 (slow) | **TF32 (fast)** | FP32 (slow) | **TF32** |
| LayerNorm / Softmax / loss computation | FP32 | FP32 | FP32 (autocast excludes these for stability) | FP32 |
| EMD loss `cumsum` | FP32 | FP32 | FP32 | FP32 |

**Takeaway**: bf16 AMP handles the hot path (conv/matmul/attention). TF32 silently accelerates everything bf16 doesn't touch. Both should be on. On RTX 5090 / 5080 Laptop with both enabled, expect ~2.5× training throughput vs naive FP32, with no accuracy degradation on IQA metrics.

**Verification**: After the first epoch, sanity-check that Tensor Core utilization is high via `nvidia-smi dmon -s u` or Nsight Systems. If you see low SM activity with high memory bandwidth, TF32/AMP is not kicking in correctly.

---

## 8. Evaluation Protocol

### Primary Metrics
- **SRCC** (Spearman Rank Correlation) — primary IQA metric, robust to monotonic transforms
- **PLCC** (Pearson Linear Correlation) — after nonlinear mapping (optional)
- **RMSE** — on normalized [0,1] scale
- **KRCC** (Kendall Rank) — supplementary

### Evaluation Modes
1. **Single-crop** (center crop): Fast sanity check during training
2. **5-crop TTA**: Final reported number for in-dataset evaluation
3. **25-crop (5×5 grid) TTA**: Optional — likely +1-2 SRCC points
4. **Cross-dataset**: Train on KonIQ-10k, evaluate on SPAQ (zero-shot, no fine-tune)

### Reporting Table Template
| Model | Params (M) | FLOPs (G) | Train Time (h) | In-dataset SRCC | In-dataset PLCC | Cross-dataset SRCC (SPAQ) |
|-------|-----------|-----------|----------------|-----------------|-----------------|---------------------------|
| Baseline CNN (MSE) | | | | | | |
| Baseline CNN (EMD) | | | | | | |
| ResNet-50 (MSE) | | | | | | |
| ResNet-50 (EMD) | | | | | | |
| ViT-B/16 LoRA (EMD) | | | | | | |

---

## 9. XAI Implementation

### 9.1 Grad-CAM (ResNet-50)
Use `pytorch-grad-cam` library:
```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

cam = GradCAM(model=resnet_model, target_layers=[model.layer4[-1]])
# Target: the bucket corresponding to predicted score
grayscale_cam = cam(input_tensor=img, targets=[ClassifierOutputTarget(pred_bucket)])
```

### 9.2 Attention Rollout (ViT-B/16)
```python
def attention_rollout(attentions: list[torch.Tensor]) -> torch.Tensor:
    """
    attentions: list of [B, num_heads, N+1, N+1] from each layer.
    Returns: [B, 14, 14] heatmap for CLS→patch attention.
    """
    result = torch.eye(attentions[0].size(-1), device=attentions[0].device)
    for attn in attentions:
        attn_heads_mean = attn.mean(dim=1)  # [B, N+1, N+1]
        attn_with_residual = attn_heads_mean + torch.eye(
            attn_heads_mean.size(-1), device=attn.device
        )
        attn_normalized = attn_with_residual / attn_with_residual.sum(dim=-1, keepdim=True)
        result = attn_normalized @ result

    # CLS token's attention to patch tokens
    cls_attention = result[:, 0, 1:]  # skip CLS-to-CLS
    return cls_attention.reshape(-1, 14, 14)
```

### 9.3 Visualization Protocol
- Select 15-20 curated test images spanning the MOS range (low/mid/high)
- Generate Grad-CAM AND Attention Rollout side-by-side
- Include original image for reference
- Save as PNG grid (4 rows × 4 cols) for the report

---

## 10. Deployment & Quantization Pipeline

### 10.0 Backend Priority (IMPORTANT — updated based on Blackwell laptop issues)

Community reports (as of April 2026) indicate that **TensorRT on RTX 50-series Laptop GPUs has a compiler bug** that can make FP16/INT8 engines 10-15× SLOWER than ONNX Runtime with CUDA backend (e.g., 107ms vs 6.4ms on ResNet on RTX 5090 Laptop). This does not affect desktop RTX 5090 as severely, but since our primary dev/test machine is the ROG laptop, **we treat ONNX Runtime as the primary deployment target**, not TensorRT.

**Deployment backend priority for this project**:
1. **ONNX Runtime + CUDAExecutionProvider** (PRIMARY) — works reliably on both 5090 desktop and 5080 Laptop
2. **ONNX Runtime + DirectMLExecutionProvider** — Windows-native, interesting for comparison (works even without CUDA)
3. **TensorRT FP16** (OPTIONAL) — run only on desktop 5090; skip on laptop if slowdown is observed
4. **TensorRT INT8** (STRETCH) — if TensorRT FP16 works cleanly, add this as the "bonus" quantization story

### 10.1 ONNX Export (`src/deploy/export_onnx.py`)
```python
import torch

model.eval()
dummy_input = torch.randn(1, 3, 224, 224, device="cuda")
torch.onnx.export(
    model, dummy_input, "resnet50_iqa.onnx",
    opset_version=17,
    input_names=["input"], output_names=["prob"],
    dynamic_axes={"input": {0: "batch"}, "prob": {0: "batch"}},
    do_constant_folding=True,
)

# Validate the exported graph
import onnx
model_onnx = onnx.load("resnet50_iqa.onnx")
onnx.checker.check_model(model_onnx)
```

### 10.1b ViT-Specific Graph Optimization (IMPORTANT)

For the ViT model, raw `torch.onnx.export` produces an extremely verbose graph — each attention block exports as 40+ primitive ops (Reshape, Transpose, MatMul, Softmax, etc.) without any fusion. Running this directly through ONNX Runtime is ~2× slower than necessary.

Microsoft's `onnxruntime-tools` provides a dedicated Transformer optimizer that fuses `LayerNorm`, `GELU`, `Attention`, `SkipLayerNorm` into single kernels. For ViT specifically, the `model_type='vit'` preset handles the CLS token and positional embedding patterns correctly.

```python
# Install: pip install onnxruntime-tools
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.fusion_options import FusionOptions

fusion_options = FusionOptions("vit")
fusion_options.enable_gelu = True
fusion_options.enable_layer_norm = True
fusion_options.enable_attention = True
fusion_options.enable_skip_layer_norm = True

optimized_model = optimizer.optimize_model(
    "vit_iqa.onnx",
    model_type="vit",
    num_heads=12,
    hidden_size=768,
    optimization_options=fusion_options,
    opt_level=99,           # Max optimization
)
optimized_model.convert_float_to_float16()  # Optional: FP16 in-place
optimized_model.save_model_to_file("vit_iqa_optimized_fp16.onnx")
```

**Expected impact**: 30-50% latency reduction for ViT on ORT CUDA backend. This is THE ViT deployment trick and absolutely worth including in the benchmark table — the ablation (raw ONNX vs optimized ONNX vs optimized+FP16) is itself a strong talking point.

**ResNet-50 note**: The Transformer optimizer does nothing for ResNet-50. Use standard ORT graph optimization instead:
```python
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession("resnet50_iqa.onnx", sess_options, providers=[...])
```

### 10.2 ONNX Runtime Benchmark (PRIMARY PATH)
```python
import onnxruntime as ort
import numpy as np

# CUDA backend
sess_cuda = ort.InferenceSession(
    "resnet50_iqa.onnx",
    providers=[
        ("CUDAExecutionProvider", {"device_id": 0, "cudnn_conv_algo_search": "EXHAUSTIVE"}),
        "CPUExecutionProvider"
    ]
)

# DirectML backend (Windows-native, no CUDA dependency)
sess_dml = ort.InferenceSession(
    "resnet50_iqa.onnx",
    providers=["DmlExecutionProvider", "CPUExecutionProvider"]
)

# Benchmark loop
dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
# Warmup
for _ in range(100):
    sess_cuda.run(None, {"input": dummy})
# Measure
import time
t0 = time.perf_counter()
for _ in range(1000):
    sess_cuda.run(None, {"input": dummy})
latency_ms = (time.perf_counter() - t0) / 1000 * 1000
print(f"ORT CUDA latency: {latency_ms:.2f} ms")
```

### 10.3 FP16 ONNX Model (for ORT FP16 inference)
```python
from onnxconverter_common import float16
import onnx

model_fp32 = onnx.load("resnet50_iqa.onnx")
model_fp16 = float16.convert_float_to_float16(model_fp32, keep_io_types=True)
onnx.save(model_fp16, "resnet50_iqa_fp16.onnx")
```

### 10.4 TensorRT Conversion (OPTIONAL — run only on desktop 5090)
```bash
# Check TensorRT availability first
python -c "import tensorrt as trt; print(trt.__version__)"

# Convert ONNX to TensorRT engine
trtexec --onnx=resnet50_iqa.onnx \
        --saveEngine=resnet50_fp16.engine \
        --fp16 \
        --shapes=input:1x3x224x224
```

**If TensorRT engine latency > ONNX Runtime CUDA latency, STOP and document the anomaly**. This is a known issue on Blackwell laptops — report the numbers as-is, don't try to force a win. The discussion itself becomes a valuable engineering finding in the report.

### 10.5 INT8 Post-Training Quantization (STRETCH GOAL)
Only attempt if ONNX Runtime or TensorRT FP16 is working smoothly.

- **Via ONNX Runtime**: Use `onnxruntime.quantization.quantize_static` with a calibration data reader
- **Via TensorRT**: Use `trtexec --int8 --calib=calibration.cache` with entropy calibrator
- **Calibration set**: 500-1000 randomly sampled images from KonIQ-10k training split
- **Validation**: Report SRCC degradation vs FP16/FP32 — <1% drop is ideal

### 10.6 Full Latency Benchmark Table (target output)

**ResNet-50 benchmark:**
| Backend | Precision | Latency (ms) | Throughput (img/s) | Model Size (MB) | SRCC | Notes |
|---------|-----------|--------------|---------------------|------------------|------|-------|
| PyTorch eager | FP32 | | | | | Baseline |
| PyTorch eager | BF16 | | | | | AMP autocast |
| PyTorch `torch.compile` | BF16 | | | | | Compiled graph |
| ONNX Runtime CUDA | FP32 | | | | | **Primary comparison point** |
| ONNX Runtime CUDA | FP16 | | | | | **Primary comparison point** |
| ONNX Runtime DirectML | FP32 | | | | | Windows-native, no CUDA dep |
| TensorRT | FP16 | | | | | Desktop 5090 only if available |
| TensorRT | INT8 | | | | | Stretch goal |

**ViT-B/16 benchmark (note the optimizer ablation):**
| Backend | Precision | Graph Opt | Latency (ms) | Throughput (img/s) | SRCC | Notes |
|---------|-----------|-----------|--------------|---------------------|------|-------|
| PyTorch eager | FP32 | N/A | | | | Baseline |
| PyTorch `torch.compile` | BF16 | N/A | | | | |
| ONNX Runtime CUDA | FP32 | Raw export | | | | Unoptimized — expect slow |
| ONNX Runtime CUDA | FP32 | ORT ALL | | | | Generic graph opt |
| ONNX Runtime CUDA | FP32 | **Transformer fusion** | | | | **Key ablation point** |
| ONNX Runtime CUDA | FP16 | Transformer fusion | | | | Best expected config |
| TensorRT | FP16 | N/A | | | | Optional |

Benchmark methodology: batch size 1, 100 warmup iterations, 1000 measured iterations, report mean ± std. Run on both machines (desktop 5090 and laptop 5080) and report both — the cross-device comparison IS the story.

---

## 11. Weekly Timeline

| Week | Milestone | Owner |
|------|-----------|-------|
| **1 (Day 1-2)** | **Environment unification across all 3 machines: PyTorch 2.9.1 + cu130, verify sm_120** | **All (critical)** |
| 1 (Day 3-7) | Data pipeline + baseline CNN training | Xuezhen |
| 2 | Baseline results + evaluation framework | Xuezhen + Haowei |
| 3 | ResNet-50 EMD implementation | Haowei |
| 4 | ResNet-50 training + Grad-CAM | Haowei |
| 5 | ViT-B/16 + LoRA setup (verify layer names first!) | Peter |
| 6 | ViT training + Attention Rollout | Peter |
| 7 (Mon-Wed) | Cross-dataset eval + systematic comparison | All |
| 7 (Thu-Sun) | ONNX Runtime benchmark + optional TensorRT | Haowei |
| 8 | Report + slides + presentation | All |

### Role Assignment
- **Haowei**: ResNet-50 + EMD + Deployment pipeline (highest technical complexity)
- **Xuezhen**: Data pipeline + Baseline CNN + Shared evaluation framework (infrastructure)
- **Peter**: ViT + LoRA + Attention Rollout (self-contained module)
- **Shared**: Evaluation metrics, visualization code, final report

---

## 12. Risk Register

| Risk | Severity | Mitigation |
|------|----------|------------|
| PyTorch < 2.9 doesn't support sm_120 (RTX 50 series) | **CRITICAL** | Lock team-wide on PyTorch 2.9.1 + cu130 (Section 2). Verify `torch.cuda.get_device_capability()` returns `(12, 0)` on Day 1. |
| Windows `spawn` + `persistent_workers` memory leak | **High** | `mp.set_start_method('spawn', force=True)` + `worker_init_fn` upfront (Section 3). |
| KonIQ-10k official server throttling / timeout | High | Kaggle mirror as primary fallback (Section 3) |
| Team members using different PyTorch/CUDA versions | High | Pin exact versions in `requirements.txt`; require `pip freeze` output in README |
| TensorRT slower than ORT CUDA on Blackwell laptop | Medium | Deployment primary path is ORT CUDA, not TensorRT (Section 10.0) |
| Raw ONNX ViT graph too slow to be meaningful | Medium | Apply ORT Transformer optimizer (Section 10.1b) |
| `torch.compile` fails on nightly builds / specific cu130 combos | Medium | Fallback to eager mode + bfloat16 AMP; re-attempt compile at Week 7 |
| KonIQ-10k download not providing per-user raw ratings | Low | Confirmed: use MOS + std for Gaussian target (Section 4) |
| ViT underperforms ResNet-50 on 10K samples | Low | Valid finding — discuss in report as "data efficiency of inductive bias" |
| TensorRT INT8 calibration too complex for timeline | Low | Skip INT8; FP16 benchmark is sufficient |
| Cross-dataset domain gap too large (SPAQ) | Low | Frame as "generalization study" — large gap itself is a finding |
| Overfitting on ViT full fine-tune | Mitigated | LoRA restricts trainable params to <1% |

---

## 13. Critical Do's and Don'ts

### DO
- Use `image_id`-based split to prevent leakage
- Clamp `std ≥ 0.5` bucket units in Gaussian target generation
- Apply softmax BEFORE feeding into EMD loss
- Use bfloat16 (not fp16) for mixed precision
- Set `RandomResizedCrop(scale=(0.5, 1.0))` — NOT default `(0.08, 1.0)`
- Use 5-crop TTA at evaluation time
- Track experiments in W&B or equivalent from Day 1

### DON'T
- Apply `GaussianBlur`, `ColorJitter`, or any quality-modifying augmentation
- Use random split on raw rows (risks same image in train+test)
- Confuse Grad-CAM (gradient-based, for CNN) with Attention Rollout (attention-based, for ViT)
- Fine-tune full ViT weights — use LoRA
- Report only single-crop results in final paper
- Skip cross-dataset evaluation — it's the most discriminating eval axis
- Forget to seed `torch.manual_seed`, `np.random.seed`, `random.seed` for reproducibility

---

## 14. Interview-Ready Talking Points

### 14.1 ML Engineer / Research Engineer Roles

When discussing this project in interviews for ML Engineer / Applied Scientist roles, these are the deepest-sharp points to lead with:

1. **EMD Loss mathematical formulation**: "We used squared Earth Mover's Distance because aesthetic ratings are ordinal — MSE on class probabilities would ignore the fact that confusing a '7' with an '8' is cheaper than confusing a '7' with a '2'."
2. **LoRA parameter efficiency**: "We reduced trainable parameters from 86M to 800K (0.9%) via LoRA on Q/V matrices, which prevented overfitting on our 10K-sample dataset while preserving pretrained knowledge."
3. **Transformer-specific graph optimization**: "We identified that raw ONNX export of ViT produces 40+ primitive ops per attention block. By applying ORT's Transformer fusion optimizer, we collapsed these into fused `Attention`, `LayerNorm`, `GELU` kernels and achieved [X]% latency reduction on top of FP16 quantization."
4. **Cross-dataset generalization**: "We observed a [X] SRCC drop from KonIQ-10k to SPAQ, highlighting that in-the-wild IQA models still carry dataset-specific biases from the annotation process."

### 14.2 Quant / HFT / Systematic Trading Roles

This project is nominally CV, but the underlying techniques map directly to Quant pain points. Frame the project this way in Quant-focused interviews:

1. **From point estimates to distributional prediction**:
   *"In this project, I didn't regress a single quality score with MSE — I predicted the full probability distribution of human ratings using EMD loss. This is conceptually identical to the quant practice of predicting return distributions rather than just expected returns, where capturing variance, skew, and tail risk matters more than the point estimate. I'm comfortable with the math of custom loss functions that work on distributions over continuous targets, and with the engineering of converting between continuous labels, discretized buckets, and calibrated output distributions."*

2. **Latency-aware deployment for time-sensitive inference**:
   *"As the deployment lead, I took the best model from PyTorch FP32 all the way through ONNX export → graph-level operator fusion → FP16 quantization → ORT CUDA backend, achieving [X]ms end-to-end inference latency on RTX 5090. In HFT and mid-frequency alpha, signal latency directly translates to PnL decay, so I'm comfortable not only training models in Python but also optimizing the inference engine layer — operator fusion, quantization calibration, and backend-specific tuning."*

3. **Systematic experimental design and cross-dataset robustness**:
   *"I designed the evaluation protocol to include cross-dataset zero-shot transfer (KonIQ-10k → SPAQ) specifically to measure out-of-distribution generalization. In quant terms, this is the difference between in-sample Sharpe and the actual live-trading Sharpe — I'm trained to be suspicious of any single-dataset benchmark number and to design evaluations that stress-test robustness."*

4. **Handling label uncertainty**:
   *"The dataset provides both MOS and standard deviation per image, reflecting annotator disagreement. Rather than throwing away that uncertainty signal, I incorporated it by constructing per-image Gaussian targets. This is analogous to modeling time-varying volatility in returns — the signal-to-noise ratio of each observation is itself information, and naive pooling wastes it."*

### 14.3 The CV → Quant Pivot in Resume Prose

When writing this project for your CV's Quant applications, emphasize the following keywords in bullet order:
- Distributional prediction (EMD loss)
- Parameter-efficient fine-tuning (LoRA)
- Inference latency optimization (ONNX, operator fusion, FP16/INT8 quantization)
- Cross-domain generalization (KonIQ-10k → SPAQ zero-shot)
- Reproducible benchmarking (batch-matched latency tests across 3 backends, 2 hardware platforms)

---

## 15. References (for report bibliography)

- Hosu et al. "KonIQ-10k: An Ecologically Valid Database for Deep Learning of Blind Image Quality Assessment." IEEE TIP, 2020.
- Hou et al. "Squared Earth Mover's Distance-based Loss for Training Deep Neural Networks." arXiv 2016.
- Talebi & Milanfar. "NIMA: Neural Image Assessment." IEEE TIP, 2018.
- Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
- Abnar & Zuidema. "Quantifying Attention Flow in Transformers." ACL 2020.
- Fang et al. "Perceptual Quality Assessment of Smartphone Photography." CVPR 2020. (SPAQ)

---

**End of Technical Specification**

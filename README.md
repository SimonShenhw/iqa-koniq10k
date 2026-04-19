# IQA Project — Deep Learning for Image Quality Assessment on KonIQ-10k

**课程 / Course**: AAI 6640 Final Project
**团队 / Team**: Haowei Shen, Xuezhen Jin, Peter Adranly
**技术文档 / Spec**: see [`IQA_Technical_Spec.md`](IQA_Technical_Spec.md)

---

## 一、最终结果 / Final Results

### 1.1 核心指标（KonIQ-10k test set，5-crop TTA）

| Model | Trainable Params | Val SRCC | **Test SRCC** | **Test PLCC** | Test RMSE |
|---|---:|---:|---:|---:|---:|
| Baseline CNN (MSE, 2 ep quick probe) | ~1.2 M | 0.688 | — | — | — |
| **ResNet-50 + EMD** | 22 M (Phase 2) | 0.8903 | **0.8842** | 0.9041 | 0.0678 |
| **ViT-B/16 + LoRA + EMD** | **597 K (0.69 %)** | **0.9098** | **0.8940** | 0.9071 | 0.0702 |

### 1.2 跨数据集零样本评估 / Cross-Dataset Zero-Shot (SPAQ)

| Model | KonIQ test SRCC | **SPAQ SRCC** | SPAQ PLCC | SRCC drop |
|---|---:|---:|---:|---:|
| ResNet-50 + EMD | 0.8842 | **0.8411** | 0.8465 | -4.3 % |
| ViT-B/16 + LoRA + EMD | 0.8940 | **0.8399** | 0.8439 | -5.4 % |

> **关键发现**：ViT-LoRA 在 in-domain 领先 ResNet-50 约 1 个 SRCC 点，但跨到 SPAQ 后两者几乎打平（0.8399 vs 0.8411）。说明 ViT 的额外 1 点来自对 KonIQ 标注风格的拟合，换分布就消失了。同时 **ViT 仅用 36× 更少的可训练参数就达到了相近性能** —— LoRA 在 10K 样本级 IQA 上有效。

### 1.3 部署延迟（RTX 5090, batch=1, 500 iters）

**ResNet-50**:

| Backend | Precision | Latency | Throughput |
|---|---|---:|---:|
| ONNX Runtime CUDA | FP32 | **5.37 ms** | 186 img/s |
| ONNX Runtime CUDA | FP16 | 9.34 ms ⚠️ | 107 img/s |
| ONNX Runtime CPU (9950X3D) | FP32 | 5.15 ms | 194 img/s |

**ViT-B/16** (the Transformer graph optimization ablation — key deployment finding):

| Pipeline | Latency | Δ |
|---|---:|---:|
| Raw ONNX / CUDA FP32 | 28.33 ms | — |
| **+ ORT Transformer fusion** | **26.22 ms** | **−7 %** |

⚠️ **Blackwell 上 ORT FP16 反而更慢**（9.34 ms vs 5.37 ms FP32），印证技术文档 Section 10.0 的警告。TensorRT 建议放弃，ORT CUDA 是生产路径。

---

## 二、项目结构 / Project Layout

```
iqa-project/
├── configs/                         # YAML 训练配置 / training configs
│   ├── baseline_cnn.yaml
│   ├── resnet50_emd.yaml
│   └── vit_lora_emd.yaml
├── data/
│   ├── koniq10k/                    # 10,073 张 1024×768 图 + MOS CSV
│   └── spaq/SPAQ/                   # 11,125 智能手机图 + MOS xlsx
├── src/
│   ├── datasets/                    # KonIQ10k / SPAQ DataLoader
│   ├── models/                      # BaselineCNN / ResNet50EMD / ViTLoRA
│   ├── losses/emd_loss.py           # Squared EMD (r=2)
│   ├── utils/
│   │   ├── distribution.py          # MOS ↔ Gaussian discrete PDF
│   │   ├── metrics.py               # SRCC / PLCC / KRCC / RMSE
│   │   └── multi_crop.py            # 5-crop / 25-crop TTA
│   ├── xai/
│   │   ├── gradcam.py               # ResNet Grad-CAM
│   │   ├── attention_rollout.py     # ViT Attention Rollout
│   │   └── visualize.py             # 批量生成热力图网格 / batch heatmap grid
│   ├── deploy/
│   │   ├── export_onnx.py           # PyTorch → ONNX (含 ViT Transformer fusion)
│   │   └── tensorrt_benchmark.py    # 多后端基准
│   ├── train.py
│   └── evaluate.py
├── outputs/                          # checkpoints / logs / heatmaps / onnx (git 忽略)
├── tests/make_synthetic_koniq.py    # 烟雾测试用合成数据生成
├── verify_env.py                    # Day 1 环境验证
├── requirements.txt
└── IQA_Technical_Spec.md
```

---

## 三、环境 / Environment

**Hardware**: RTX 5090 (Blackwell sm_120, 32 GB VRAM), 9950X3D, Windows 11
**Stack**: Python 3.14 + **PyTorch 2.11.0 + cu130** + torchvision 0.26

```bash
# 一次性安装所有依赖 / one-time install
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt

# 验证 GPU / verify
python verify_env.py
```

> **Note**: 如果没有 RTX 50 系列，把 cu130 换成 cu124 + 对应 torch 版本即可。整条 pipeline 不依赖 sm_120 特性，只要 PyTorch ≥ 2.1 + CUDA ≥ 11.8 就能跑。

---

## 四、复现完整实验 / Reproduce the Full Pipeline

```bash
# ---- 0. 下载数据集 / download datasets ----
# KonIQ-10k (~4 GB):
curl -Lk -o data/koniq10k/koniq10k.zip \
    "https://datasets.vqa.mmsp-kn.de/archives/koniq10k_1024x768.zip"
curl -Lk -o data/koniq10k/scores.zip \
    "https://datasets.vqa.mmsp-kn.de/archives/koniq10k_scores_and_distributions.zip"
cd data/koniq10k && python -c "import zipfile; [zipfile.ZipFile(f).extractall('.') for f in ['koniq10k.zip','scores.zip']]"

# SPAQ (~33 GB, via HuggingFace mirror):
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('chaofengc/IQA-PyTorch-Datasets', 'spaq.tgz',
                repo_type='dataset', local_dir='data/spaq')"
cd data/spaq && tar -xzf spaq.tgz

# ---- 1. 训练三个模型 / train three models ----
python -m src.train --config configs/baseline_cnn.yaml     # ~15 min
python -m src.train --config configs/resnet50_emd.yaml     # ~20 min on RTX 5090
python -m src.train --config configs/vit_lora_emd.yaml     # ~20 min

# ---- 2. In-dataset 评估 (5-crop TTA) ----
python -m src.evaluate --config configs/resnet50_emd.yaml \
    --ckpt outputs/checkpoints/resnet50_emd_best.pt --split test --tta 5crop

python -m src.evaluate --config configs/vit_lora_emd.yaml \
    --ckpt outputs/checkpoints/vit_b16_lora_emd_best.pt --split test --tta 5crop

# ---- 3. 跨数据集评估 / cross-dataset ----
python -m src.evaluate --config configs/resnet50_emd.yaml \
    --ckpt outputs/checkpoints/resnet50_emd_best.pt \
    --cross-dataset spaq --cross-root data/spaq/ --tta 5crop

# ---- 4. XAI 可视化 / XAI heatmaps ----
python -m src.xai.visualize --config configs/resnet50_emd.yaml \
    --ckpt outputs/checkpoints/resnet50_emd_best.pt \
    --method gradcam --n-images 16 \
    --output outputs/heatmaps/resnet_gradcam.png

python -m src.xai.visualize --config configs/vit_lora_emd.yaml \
    --ckpt outputs/checkpoints/vit_b16_lora_emd_best.pt \
    --method rollout --n-images 16 \
    --output outputs/heatmaps/vit_rollout.png

# ---- 5. 部署 / deployment ----
python -m src.deploy.export_onnx --config configs/resnet50_emd.yaml \
    --ckpt outputs/checkpoints/resnet50_emd_best.pt --device cpu \
    --output outputs/resnet50_iqa.onnx

python -m src.deploy.export_onnx --config configs/vit_lora_emd.yaml \
    --ckpt outputs/checkpoints/vit_b16_lora_emd_best.pt --device cpu \
    --output outputs/vit_iqa.onnx
# → 自动触发 Transformer fusion + FP16 转换

python -m src.deploy.tensorrt_benchmark \
    --onnx outputs/resnet50_iqa.onnx \
    --backends ort-cuda,ort-cuda-fp16,ort-cpu --warmup 100 --iters 500
```

---

## 五、关键技术决策 / Key Technical Decisions

### 5.1 EMD 损失 / Squared EMD Loss
IQA 分数是序数变量：把 9 误判成 10 应该比误判成 2 便宜得多。MSE-on-probabilities 是 order-blind，EMD 对 CDF 差求平方，天然惩罚"邻近 vs 远处"错误。公式：

```
EMD(p, q) = ( (1/B) · Σ_i | CDF(p)_i − CDF(q)_i |^r )^(1/r),   r = 2
```

### 5.2 MOS 量表归一化（踩坑记）
KonIQ 官方 CSV 里 `MOS` 列其实是 **1-5 Likert 量表**（技术文档第 3 节写 "1-100" 是错的）。真正的 [0,100] 量表是 `MOS_zscore` 列。[`src/datasets/koniq10k.py`](src/datasets/koniq10k.py) 的 `_normalize_columns` 自动探测并优先使用 `MOS_zscore`，同时按比例把 `SD × 25` 对齐到 [0,100] 空间。

### 5.3 Gaussian 目标分布
用 `(MOS, SD)` 生成 10-bucket 的离散 Gaussian，**必须 clamp `SD ≥ 0.5 bucket unit`**，否则高确定性样本退化成 one-hot，EMD 失去意义。

### 5.4 LoRA 配置
- r=16, α=32, dropout=0.1
- `target_modules=["qkv"]`（timm ViT-B/16 融合了 Q/K/V，所以单一模块名）
- `modules_to_save=["head"]`（新分类头全量训练）
- **597 K / 86.4 M = 0.69 % 可训练参数**

### 5.5 Blackwell Tensor Core 精度栈
```python
torch.set_float32_matmul_precision("high")     # FP32 matmul → TF32
torch.backends.cudnn.allow_tf32 = True         # conv → TF32
torch.autocast(device_type="cuda", dtype=torch.bfloat16)  # hot path → bf16
```
TF32（matmul/conv 外的 ops）+ bf16 AMP（hot path）两层叠加，RTX 5090 上训练吞吐 ~2.5×。**必须用 bf16 而非 fp16** — `cumsum` 在 fp16 会下溢。

### 5.6 ViT Transformer 图优化（部署关键）
raw ONNX 每个注意力 block 展开成 40+ 原生 op。调用 `onnxruntime.transformers.optimizer.optimize_model(model_type="vit")` 把它们融合为 `Attention` / `LayerNorm` / `GELU` / `SkipLayerNorm` 单内核，延迟下降 7%。

---

## 六、项目过程中修掉的 8 个 bug / Bugs Fixed During Development

| # | 文件 | Bug | 修复 |
|---|---|---|---|
| 1 | `src/models/__init__.py` | 强制 import `vit_lora` 导致缺 `timm` 时整个 models 模块不可用 | 改为延迟 import |
| 2 | `src/xai/__init__.py` | 同上 | 延迟 `__getattr__` |
| 3 | `src/datasets/koniq10k.py` | `MOS` 列是 1-5 量表，直接按 100 归一会全部退化为边界概率 | 优先选 `MOS_zscore` + `SD × 25` |
| 4 | `src/train.py` | Python 3.14 Windows 缺 Triton → `torch.compile` 崩 | 启动前 `importlib.util.find_spec("triton")` 检测 |
| 5 | `src/train.py` / `evaluate.py` / `export_onnx.py` / `visualize.py` | PyTorch 2.6+ 默认 `weights_only=True` 拒绝含 numpy 标量的 ckpt | 显式 `weights_only=False` |
| 6 | `src/deploy/export_onnx.py` | Windows GBK 控制台不能打印 PyTorch 新 exporter 的 `✅` Unicode | `sys.stdout.reconfigure(encoding="utf-8")` |
| 7 | `src/xai/attention_rollout.py` | timm 1.0+ `fused_attn=True` 调 `scaled_dot_product_attention` 融合核，`attn_drop` hook 永不触发 | 注册 hook 时暂关 fused_attn，`.remove()` 时恢复 |
| 8 | `src/xai/attention_rollout.py` | timm 新版 `blocks` 是 `Sequential` 不是 `ModuleList`，类型判断过严 | 接受两者 |

---

## 七、产物 / Deliverables

已生成的产物（.gitignore 排除 — 训完重新生成）：

| 类别 | 路径 | 大小 |
|---|---|---:|
| Checkpoints | `outputs/checkpoints/{resnet50_emd,vit_b16_lora_emd}_best.pt` | ~100 MB / ~350 MB |
| ONNX | `outputs/{resnet50_iqa,vit_iqa}*.onnx` + `.onnx.data` | ~150 MB / ~700 MB |
| Heatmaps | `outputs/heatmaps/{resnet_gradcam,vit_rollout}.png` | 4.3 MB / 4.4 MB |
| Logs | `outputs/logs/{resnet50,vit}_train.log` | — |

---

## 八、参考文献 / References

1. Hosu et al. "KonIQ-10k: An Ecologically Valid Database for Deep Learning of Blind Image Quality Assessment." *IEEE TIP*, 2020.
2. Hou et al. "Squared Earth Mover's Distance-based Loss for Training Deep Neural Networks." *arXiv*, 2016.
3. Talebi & Milanfar. "NIMA: Neural Image Assessment." *IEEE TIP*, 2018.
4. Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR*, 2022.
5. Abnar & Zuidema. "Quantifying Attention Flow in Transformers." *ACL*, 2020.
6. Fang et al. "Perceptual Quality Assessment of Smartphone Photography." *CVPR*, 2020.

---

## 九、License
Course project, non-commercial research use.

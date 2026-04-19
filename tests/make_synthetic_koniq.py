"""
生成最小合成 KonIQ-10k 数据 / Generate a tiny synthetic KonIQ-10k stub.

用途 / Purpose:
    在真实数据集下载前，做端到端训练/评估烟雾测试。
    For end-to-end pipeline smoke testing before the real dataset arrives.

生成内容 / What it creates:
    data/koniq10k_synthetic/
        ├── 1024x768/  (100 张 1024x768 的随机 JPEG)
        └── koniq10k_scores_and_distributions.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def make_stub(root: Path, n: int = 100, size=(1024, 768)) -> None:
    img_dir = root / "1024x768"
    img_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        arr = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
        name = f"img_{i:04d}.jpg"
        Image.fromarray(arr).save(img_dir / name, quality=85)
        mos = float(rng.uniform(20.0, 90.0))
        sd = float(rng.uniform(3.0, 15.0))
        rows.append({"image_name": name, "MOS": mos, "SD": sd})

    df = pd.DataFrame(rows)
    df.to_csv(root / "koniq10k_scores_and_distributions.csv", index=False)
    print(f"[OK] Generated {n} synthetic images at {root}")


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/koniq10k_synthetic")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    make_stub(root, n=n)

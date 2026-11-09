"""eval/run_ablation.py
Ablation experiment suite for FabricQA_CV Phase 2 (Texture Inspector).

Evaluates texture configurations:
1. Baseline (Phase 1: Edge + Spectral + Seam, no texture)
2. LBP + Entropy only
3. LBP + Entropy + Gabor Bank (18 filters)
4. LBP + Entropy + GLCM Haralick Features (32 levels)
5. All Combined (LBP + Entropy + Gabor + GLCM)

Computes:
- Image-level AUROC
- Image-level F1 score
- False Positive Rate (FPR) on clean fabrics
- Mean latency per 4096x256 fabric strip
- 1024x1024 standard image processing latency
"""

import os
import sys
import io
import time
import argparse
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.run_eval import discover_aitex_strips, load_and_orient_strip
from inspectors.unified_processor import UnifiedProcessor, unified_processor


def benchmark_1024(feature_mode: str, runs: int = 5) -> float:
    """Benchmark end-to-end 1024x1024 image processing latency for a given texture mode."""
    h, w = 1024, 1024
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)
    base = 128 + 25 * np.sin(2 * np.pi * xx / 10.0) + 25 * np.cos(2 * np.pi * yy / 10.0)
    noise = np.random.normal(0, 4, (h, w))
    img = cv2.cvtColor(np.clip(base + noise, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    _, enc = cv2.imencode(".png", img)
    buf = io.BytesIO(enc.tobytes())

    # Warmup
    _ = unified_processor.process(io.BytesIO(enc.tobytes()), feature_mode=feature_mode)

    latencies = []
    for _ in range(runs):
        b = io.BytesIO(enc.tobytes())
        t0 = time.perf_counter()
        _ = unified_processor.process(b, feature_mode=feature_mode)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(latencies))


def evaluate_mode(
    strips: List[Dict[str, Any]],
    feature_mode: str,
    tile_w: int = 1024,
    sensitivity: float = 2.5,
    quality_threshold: int = 30,
) -> Dict[str, Any]:
    """Evaluate a single feature mode across the strip collection."""
    print(f"\n---> Evaluating mode: '{feature_mode}' across {len(strips)} strips...")
    t0_all = time.time()

    records = []
    strip_latencies = []

    for idx, s in enumerate(strips):
        img, _ = load_and_orient_strip(s["img_path"], s["mask_paths"])
        if img is None:
            continue

        h, w = img.shape[:2]
        num_tiles = max(1, w // tile_w)
        strip_defects = []
        strip_t0 = time.perf_counter()

        for t_idx in range(num_tiles):
            x1 = t_idx * tile_w
            x2 = min(w, (t_idx + 1) * tile_w)
            tile = img[:, x1:x2]

            ok, enc = cv2.imencode(".png", tile)
            if not ok:
                continue

            buf = io.BytesIO(enc.tobytes())
            res = unified_processor.process(buf, sensitivity=sensitivity, feature_mode=feature_mode)
            for d in res.get("defects", []):
                strip_defects.append(d)

        strip_ms = (time.perf_counter() - strip_t0) * 1000.0
        strip_latencies.append(strip_ms)

        # Confidence calculation
        def _conf(d):
            c = d.get("Quality Score", "0%")
            try:
                return int(str(c).replace("%", "").strip())
            except Exception:
                return 0

        filtered = [d for d in strip_defects if _conf(d) >= quality_threshold]
        is_detected = len(filtered) > 0
        max_score = max([_conf(d) / 100.0 for d in filtered], default=0.0)

        records.append({
            "id": s["id"],
            "ground_truth": s["ground_truth"],
            "pred_detected": 1 if is_detected else 0,
            "max_score": max_score,
            "defect_count": len(filtered),
        })

    y_true = [r["ground_truth"] for r in records]
    y_pred = [r["pred_detected"] for r in records]
    y_scores = [r["max_score"] for r in records]

    try:
        auroc = float(roc_auc_score(y_true, y_scores)) if len(np.unique(y_true)) > 1 else 0.0
    except Exception:
        auroc = 0.0

    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / max(tn + fp, 1))
    mean_strip_ms = float(np.mean(strip_latencies)) if strip_latencies else 0.0

    bench_1024_ms = benchmark_1024(feature_mode, runs=5)

    print(f"     Done in {time.time() - t0_all:.1f}s | AUROC: {auroc:.4f} | F1: {f1:.4f} | FPR: {fpr:.4f} | 1024x1024: {bench_1024_ms:.1f}ms")

    return {
        "mode": feature_mode,
        "auroc": round(auroc, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "mean_strip_ms": round(mean_strip_ms, 2),
        "bench_1024_ms": round(bench_1024_ms, 2),
    }


def run_ablation(dataset_dir: str = "data/aitex", max_strips: Optional[int] = 60):
    strips = discover_aitex_strips(dataset_dir)
    if not strips:
        raise FileNotFoundError(f"No AITEX strips found in {dataset_dir}")

    # Subsample balanced subset for fast ablation if requested
    if max_strips is not None and max_strips < len(strips):
        defs = [s for s in strips if s["ground_truth"] == 1]
        cleans = [s for s in strips if s["ground_truth"] == 0]
        n_def = min(len(defs), max_strips // 2)
        n_clean = min(len(cleans), max_strips - n_def)
        eval_strips = defs[:n_def] + cleans[:n_clean]
        print(f"[*] Subsampling evaluation to {len(eval_strips)} strips ({n_def} defective, {n_clean} clean)")
    else:
        eval_strips = strips
        print(f"[*] Evaluating full dataset of {len(eval_strips)} fabric strips")

    modes = [
        ("Phase 1 Baseline (No Texture)", "none"),
        ("LBP + Local Entropy Only", "lbp_entropy"),
        ("+ Gabor Filter Bank (18 filters)", "gabor"),
        ("+ GLCM Haralick Features (32 levels)", "glcm"),
        ("All Combined (LBP + Gabor + GLCM)", "all"),
    ]

    results = []
    for label, mode_key in modes:
        res = evaluate_mode(eval_strips, mode_key)
        res["label"] = label
        results.append(res)

    df = pd.DataFrame(results)
    df.to_csv("eval/phase2_ablation.csv", index=False)

    md = f"""# FabricQA_CV - Phase 2 Texture Ablation Study

**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Dataset:** AITEX Fabric Database ({len(eval_strips)} evaluated strips)  
**Tiling Strategy:** 1024x256 horizontal patches  

## Feature Bank Ablation Results

| Configuration | Feature Mode | AUROC | F1 Score | False Positive Rate (FPR) | Strip Latency (4096x256) | 1024x1024 Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for r in results:
        md += f"| **{r['label']}** | `{r['mode']}` | **{r['auroc']:.4f}** | **{r['f1']:.4f}** | **{r['fpr']:.4f}** | {r['mean_strip_ms']:.1f} ms | {r['bench_1024_ms']:.1f} ms |\n"

    md += """
## Confusion Matrix Breakdown

| Configuration | TP | FP | TN | FN |
|:---|:---:|:---:|:---:|:---:|
"""
    for r in results:
        md += f"| {r['label']} | {r['tp']} | {r['fp']} | {r['tn']} | {r['fn']} |\n"

    with open("eval/phase2_ablation.md", "w", encoding="utf-8") as f:
        f.write(md)

    print("\n" + "=" * 60)
    print(" Ablation Study Complete. Report written to eval/phase2_ablation.md")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 2 Texture Ablation")
    parser.add_argument("--data-dir", type=str, default="data/aitex")
    parser.add_argument("--max-strips", type=int, default=60)
    args = parser.parse_args()

    run_ablation(dataset_dir=args.data_dir, max_strips=args.max_strips)


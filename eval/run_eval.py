"""eval/run_eval.py
Evaluation harness for FabricQA_CV on the AITEX fabric defect benchmark.

Dataset structure:
- data/aitex/Defect_images: 106 defective fabric strips (4096x256, 12 defect types)
- data/aitex/Mask_images: 107 ground-truth binary masks (*_mask*.png)
- data/aitex/NODefect_images: 141 defect-free fabric strips across 7 fabrics

Metrics:
- Image-level AUROC & F1 score
- False Positive Rate (FPR) on defect-free fabrics
- Pixel-level Mean IoU & Pixel-level AUROC on defective fabrics
- Per-strip execution time & 1024x1024 standard latency benchmark

Outputs:
- eval/baseline_results.csv
- eval/baseline.md
"""

import os
import sys
import io
import glob
import time
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

from inspectors.unified_processor import unified_processor, UnifiedProcessor


def discover_aitex_strips(dataset_dir: str) -> List[Dict[str, Any]]:
    """Discover all defective and defect-free strips in the AITEX directory."""
    defect_dir = os.path.join(dataset_dir, "Defect_images")
    mask_dir = os.path.join(dataset_dir, "Mask_images")
    nodefect_dir = os.path.join(dataset_dir, "NODefect_images")

    strips = []

    # 1. Defective strips
    if os.path.exists(defect_dir):
        def_files = sorted(glob.glob(os.path.join(defect_dir, "*.png")))
        for f in def_files:
            base = os.path.splitext(os.path.basename(f))[0]
            mask_files = sorted(glob.glob(os.path.join(mask_dir, f"{base}_mask*.png")))
            strips.append({
                "id": base,
                "img_path": f,
                "mask_paths": mask_files,
                "ground_truth": 1,
                "category": "Defective",
            })

    # 2. Defect-free strips
    if os.path.exists(nodefect_dir):
        nodef_files = sorted(glob.glob(os.path.join(nodefect_dir, "**", "*.png"), recursive=True))
        for f in nodef_files:
            fabric_name = os.path.basename(os.path.dirname(f))
            base = f"{fabric_name}_{os.path.splitext(os.path.basename(f))[0]}"
            strips.append({
                "id": base,
                "img_path": f,
                "mask_paths": [],
                "ground_truth": 0,
                "category": "DefectFree",
            })

    return strips


def load_and_orient_strip(img_path: str, mask_paths: List[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load fabric strip and ground truth mask, normalizing to horizontal 256x4096 orientation."""
    img = cv2.imread(img_path)
    if img is None:
        return None, None

    if img.shape[0] > img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    h, w = img.shape[:2]
    combined_mask = None

    if mask_paths:
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for mp in mask_paths:
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            if m.shape[0] > m.shape[1]:
                m = cv2.rotate(m, cv2.ROTATE_90_CLOCKWISE)
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            combined_mask = np.bitwise_or(combined_mask, m)

    return img, combined_mask


def evaluate_single_strip(
    strip_info: Dict[str, Any],
    processor: UnifiedProcessor,
    tile_w: int = 1024,
    sensitivity: float = 2.5,
    quality_threshold: int = 30,
) -> Dict[str, Any]:
    """Process a 4096x256 strip by tiling, evaluating tiles, and aggregating results."""
    img, gt_mask = load_and_orient_strip(strip_info["img_path"], strip_info["mask_paths"])
    if img is None:
        return {"error": "Failed to load image"}

    h, w = img.shape[:2]
    num_tiles = max(1, w // tile_w)
    all_strip_defects = []
    strip_latency_ms = 0.0

    # Process each tile
    for t_idx in range(num_tiles):
        x1 = t_idx * tile_w
        x2 = min(w, (t_idx + 1) * tile_w)
        tile = img[:, x1:x2]

        success, enc = cv2.imencode(".png", tile)
        if not success:
            continue
        buf = io.BytesIO(enc.tobytes())

        t0 = time.perf_counter()
        res = processor.process(buf, sensitivity=sensitivity, mode="full", remove_shadows=False)
        strip_latency_ms += (time.perf_counter() - t0) * 1000.0

        for d in res.get("defects", []):
            d_copy = dict(d)
            # Offset x-coordinate back to global strip coordinates
            d_copy["bbox_x"] = d_copy.get("bbox_x", 0) + x1
            all_strip_defects.append(d_copy)

    # Filter by quality score
    def _conf(d):
        c = d.get("Quality Score", "0%")
        try:
            return int(str(c).replace("%", "").strip())
        except Exception:
            return 0

    filtered_defects = [d for d in all_strip_defects if _conf(d) >= quality_threshold]

    # Strip-level prediction
    is_detected = len(filtered_defects) > 0
    max_score = max([_conf(d) / 100.0 for d in filtered_defects], default=0.0)

    # Pixel-level metrics on defective samples
    iou = None
    pixel_auroc = None

    if gt_mask is not None and strip_info["ground_truth"] == 1:
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        for d in filtered_defects:
            bx = max(0, min(w - 1, int(d.get("bbox_x", 0))))
            by = max(0, min(h - 1, int(d.get("bbox_y", 0))))
            bw = max(1, min(w - bx, int(d.get("bbox_w", 0))))
            bh = max(1, min(h - by, int(d.get("bbox_h", 0))))
            pred_mask[by:by + bh, bx:bx + bw] = 255

        gt_bin = (gt_mask > 0).astype(np.uint8)
        pred_bin = (pred_mask > 0).astype(np.uint8)

        inter = np.count_nonzero(gt_bin & pred_bin)
        union = np.count_nonzero(gt_bin | pred_bin)
        iou = float(inter / max(union, 1))

        if np.any(gt_bin > 0) and not np.all(gt_bin > 0):
            try:
                # Subsample (1 in 8 px) for speed across 4096x256
                sub_gt = gt_bin[::2, ::8].ravel()
                sub_pred = pred_bin[::2, ::8].ravel()
                if len(np.unique(sub_gt)) > 1 and len(np.unique(sub_pred)) > 1:
                    pixel_auroc = float(roc_auc_score(sub_gt, sub_pred))
                else:
                    pixel_auroc = 0.5 if len(np.unique(sub_pred)) == 1 else 0.0
            except Exception:
                pixel_auroc = None

    return {
        "id": strip_info["id"],
        "category": strip_info["category"],
        "ground_truth": strip_info["ground_truth"],
        "pred_detected": 1 if is_detected else 0,
        "max_score": round(max_score, 4),
        "defect_count": len(filtered_defects),
        "latency_ms": round(strip_latency_ms, 2),
        "pixel_iou": round(iou, 4) if iou is not None else None,
        "pixel_auroc": round(pixel_auroc, 4) if pixel_auroc is not None else None,
    }


def benchmark_1024x1024(processor: Optional[UnifiedProcessor] = None, runs: int = 10) -> Dict[str, float]:
    """Measure latency on a standard 1024x1024 fabric texture image across multiple runs."""
    if processor is None:
        processor = unified_processor

    h, w = 1024, 1024
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)
    base = 128 + 25 * np.sin(2 * np.pi * xx / 10.0) + 25 * np.cos(2 * np.pi * yy / 10.0)
    noise = np.random.normal(0, 4, (h, w))
    img_1024 = cv2.cvtColor(np.clip(base + noise, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    success, enc = cv2.imencode(".png", img_1024)
    buf = io.BytesIO(enc.tobytes())

    # Warmup
    _ = processor.process(io.BytesIO(enc.tobytes()))

    latencies = []
    for _ in range(runs):
        b = io.BytesIO(enc.tobytes())
        t0 = time.perf_counter()
        _ = processor.process(b)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "mean_ms": round(float(np.mean(latencies)), 2),
        "std_ms": round(float(np.std(latencies)), 2),
        "min_ms": round(float(np.min(latencies)), 2),
        "max_ms": round(float(np.max(latencies)), 2),
    }


def run_benchmark(
    dataset_dir: str = "data/aitex",
    output_md: str = "eval/baseline.md",
    output_csv: str = "eval/baseline_results.csv",
    tile_w: int = 1024,
    max_strips: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute complete baseline evaluation on AITEX dataset."""
    os.makedirs(os.path.dirname(output_md) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    print("=================================================================")
    print(" FabricQA_CV - Industrial Evaluation & Benchmark Suite")
    print("=================================================================")

    # 1. 1024x1024 Latency Benchmark
    print("[*] Running 1024x1024 standard image benchmark (10 iterations)...")
    bench_1024 = benchmark_1024x1024()
    print(f"    -> Latency: {bench_1024['mean_ms']} ms ± {bench_1024['std_ms']} ms "
          f"(min: {bench_1024['min_ms']} ms, max: {bench_1024['max_ms']} ms)")

    # 2. Discover AITEX strips
    strips = discover_aitex_strips(dataset_dir)
    if not strips:
        raise FileNotFoundError(f"No AITEX strips found in {dataset_dir}")

    # Stratified balance if max_strips is requested
    if max_strips is not None and max_strips < len(strips):
        defs = [s for s in strips if s["ground_truth"] == 1]
        cleans = [s for s in strips if s["ground_truth"] == 0]
        n_def = min(len(defs), max_strips // 2)
        n_clean = min(len(cleans), max_strips - n_def)
        strips = defs[:n_def] + cleans[:n_clean]
        print(f"[*] Subsampling evaluation to {len(strips)} strips ({n_def} defective, {n_clean} clean)")
    else:
        print(f"[*] Evaluating full dataset of {len(strips)} fabric strips "
              f"({sum(1 for s in strips if s['ground_truth'] == 1)} defective, "
              f"{sum(1 for s in strips if s['ground_truth'] == 0)} clean)")

    # 3. Run evaluation across all strips
    records = []
    t_start = time.time()
    for idx, s in enumerate(strips):
        res = evaluate_single_strip(s, unified_processor, tile_w=tile_w)
        records.append(res)
        if (idx + 1) % 10 == 0 or (idx + 1) == len(strips):
            print(f"    Processed {idx + 1}/{len(strips)} strips... "
                  f"(Elapsed: {time.time() - t_start:.1f}s)")

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"[✓] Saved per-strip CSV results to: {output_csv}")

    # 4. Compute Metrics
    y_true = df["ground_truth"].tolist()
    y_pred = df["pred_detected"].tolist()
    y_scores = df["max_score"].tolist()

    try:
        auroc = float(roc_auc_score(y_true, y_scores)) if len(np.unique(y_true)) > 1 else 0.0
    except Exception:
        auroc = 0.0

    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / max(tn + fp, 1))

    valid_ious = df["pixel_iou"].dropna().tolist()
    valid_p_aurocs = df["pixel_auroc"].dropna().tolist()
    mean_iou = float(np.mean(valid_ious)) if valid_ious else 0.0
    mean_p_auroc = float(np.mean(valid_p_aurocs)) if valid_p_aurocs else 0.0
    mean_strip_latency = float(df["latency_ms"].mean())

    summary = {
        "total_strips": len(df),
        "defective_strips": int(sum(y_true)),
        "clean_strips": int(len(y_true) - sum(y_true)),
        "image_auroc": round(auroc, 4),
        "image_f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "pixel_iou": round(mean_iou, 4),
        "pixel_auroc": round(mean_p_auroc, 4),
        "mean_strip_latency_ms": round(mean_strip_latency, 2),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "bench_1024": bench_1024,
    }

    # 5. Output Markdown Summary
    md_content = f"""# FabricQA_CV - Evaluation Baseline

**Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Branch:** `feature/enhancements-v2` (Phase 0 Baseline)  
**Dataset:** AITEX Fabric Image Database  
**Total Evaluated Strips:** {summary['total_strips']} ({summary['defective_strips']} Defective, {summary['clean_strips']} Defect-free)  

## Key Performance Indicators (KPIs)

| Metric | Baseline Value | Description |
|:---|:---:|:---|
| **Image-level AUROC** | **{summary['image_auroc']:.4f}** | Area under ROC curve using defect quality confidence scores |
| **Image-level F1 Score** | **{summary['image_f1']:.4f}** | Harmonic mean of precision and recall (operating threshold) |
| **False Positive Rate (FPR)** | **{summary['false_positive_rate']:.4f}** | False alarms on defect-free fabrics ({summary['fp']}/{summary['clean_strips']}) |
| **Pixel-level Mean IoU** | **{summary['pixel_iou']:.4f}** | Intersection-over-Union against ground truth defect masks |
| **Pixel-level AUROC** | **{summary['pixel_auroc']:.4f}** | Pixel classification saliency AUROC across defective masks |
| **Mean Strip Latency (4096x256)** | **{summary['mean_strip_latency_ms']:.2f} ms** | End-to-end processing time per 4096x256 fabric strip |
| **1024x1024 Standard Latency** | **{bench_1024['mean_ms']:.2f} ms ± {bench_1024['std_ms']:.2f} ms** | Benchmark over 10 runs on 1024x1024 fabric texture |

## Confusion Matrix

- **True Positives (TP):** {summary['tp']}
- **False Positives (FP):** {summary['fp']}
- **True Negatives (TN):** {summary['tn']}
- **False Negatives (FN):** {summary['fn']}

## Methodology & Tiling Notes
- **AITEX Line-Scan Aspect Ratio (4096x256):** Direct global downsampling of a 4096x256 strip to width 800 collapses the height to 50 pixels, flattening woven structures and causing morphological minimum-area filters to discard real defects. The evaluation harness partitions each strip into four 1024x256 tiles, processes each tile through the unified detection pipeline, and maps detected bounding boxes back to global coordinates.
- **Operating Parameters:** Default `sensitivity=2.5`, `quality_threshold=30%`, with overlap deduplication.
"""
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"[✓] Saved baseline report to: {output_md}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate FabricQA_CV on AITEX Dataset")
    parser.add_argument("--data-dir", type=str, default="data/aitex")
    parser.add_argument("--output-md", type=str, default="eval/baseline.md")
    parser.add_argument("--output-csv", type=str, default="eval/baseline_results.csv")
    parser.add_argument("--tile-w", type=int, default=1024)
    parser.add_argument("--max-strips", type=int, default=None)
    args = parser.parse_args()

    run_benchmark(
        dataset_dir=args.data_dir,
        output_md=args.output_md,
        output_csv=args.output_csv,
        tile_w=args.tile_w,
        max_strips=args.max_strips,
    )


if __name__ == "__main__":
    main()

# FabricQA_CV - Evaluation Baseline

**Evaluation Date:** 2026-11-09 18:35:26  
**Branch:** `feature/enhancements-v2` (Phase 0 Baseline)  
**Dataset:** AITEX Fabric Image Database  
**Total Evaluated Strips:** 60 (30 Defective, 30 Defect-free)  

## Key Performance Indicators (KPIs)

| Metric | Baseline Value | Description |
|:---|:---:|:---|
| **Image-level AUROC** | **0.6272** | Area under ROC curve using defect quality confidence scores |
| **Image-level F1 Score** | **0.6667** | Harmonic mean of precision and recall (operating threshold) |
| **False Positive Rate (FPR)** | **1.0000** | False alarms on defect-free fabrics (30/30) |
| **Pixel-level Mean IoU** | **0.0036** | Intersection-over-Union against ground truth defect masks |
| **Pixel-level AUROC** | **0.4049** | Pixel classification saliency AUROC across defective masks |
| **Mean Strip Latency (4096x256)** | **1733.38 ms** | End-to-end processing time per 4096x256 fabric strip |
| **1024x1024 Standard Latency** | **1337.10 ms ± 68.29 ms** | Benchmark over 10 runs on 1024x1024 fabric texture |

## Confusion Matrix

- **True Positives (TP):** 30
- **False Positives (FP):** 30
- **True Negatives (TN):** 0
- **False Negatives (FN):** 0

## Methodology & Tiling Notes
- **AITEX Line-Scan Aspect Ratio (4096x256):** Direct global downsampling of a 4096x256 strip to width 800 collapses the height to 50 pixels, flattening woven structures and causing morphological minimum-area filters to discard real defects. The evaluation harness partitions each strip into four 1024x256 tiles, processes each tile through the unified detection pipeline, and maps detected bounding boxes back to global coordinates.
- **Operating Parameters:** Default `sensitivity=2.5`, `quality_threshold=30%`, with overlap deduplication.

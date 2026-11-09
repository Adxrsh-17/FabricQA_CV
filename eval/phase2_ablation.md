# FabricQA_CV - Phase 2 Texture Ablation Study

**Date:** 2026-11-09 19:09:59  
**Dataset:** AITEX Fabric Database (60 evaluated strips)  
**Tiling Strategy:** 1024x256 horizontal patches  

## Feature Bank Ablation Results

| Configuration | Feature Mode | AUROC | F1 Score | False Positive Rate (FPR) | Strip Latency (4096x256) | 1024x1024 Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Phase 1 Baseline (No Texture)** | `none` | **0.6272** | **0.6667** | **1.0000** | 1714.4 ms | 4493.4 ms |
| **LBP + Local Entropy Only** | `lbp_entropy` | **0.6394** | **0.6667** | **1.0000** | 2318.8 ms | 7022.4 ms |
| **+ Gabor Filter Bank (18 filters)** | `gabor` | **0.6394** | **0.6667** | **1.0000** | 3232.5 ms | 11254.1 ms |
| **+ GLCM Haralick Features (32 levels)** | `glcm` | **0.7100** | **0.6667** | **1.0000** | 6095.4 ms | 9784.9 ms |
| **All Combined (LBP + Gabor + GLCM)** | `all` | **0.6422** | **0.6667** | **1.0000** | 6370.6 ms | 11575.3 ms |

## Confusion Matrix Breakdown

| Configuration | TP | FP | TN | FN |
|:---|:---:|:---:|:---:|:---:|
| Phase 1 Baseline (No Texture) | 30 | 30 | 0 | 0 |
| LBP + Local Entropy Only | 30 | 30 | 0 | 0 |
| + Gabor Filter Bank (18 filters) | 30 | 30 | 0 | 0 |
| + GLCM Haralick Features (32 levels) | 30 | 30 | 0 | 0 |
| All Combined (LBP + Gabor + GLCM) | 30 | 30 | 0 | 0 |

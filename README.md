# Fabric Defect Detector

**Automated Optical Inspection (AOI) System for Fabric Defect Detection** *Built with Classical Computer Vision & Python*

![Project Status](https://img.shields.io/badge/Status-Active-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-orange)

## Overview
**FabricQA** is a robust, reference-free inspection tool designed to detect manufacturing defects in textile fabrics. Unlike modern AI approaches that require thousands of training images, this system uses **Classical Computer Vision** techniques (LBP, Gabor Filters, FFT Spectral Analysis, Morphological Processing, and Statistical Profiling) to detect defects mathematically.

It can identify and classify:
* **Holes** (Punctures, Ragged Holes)
* **Oil/Water Stains** (Discoloration)
* **Cuts & Tears** (Structural damage — horizontal and vertical)
* **Snags / Knots** (Texture anomalies)
* **Skip Stitches** (Seam defects)
* **Wrinkles / Folds** (Structural deformation)
* **Weave Irregularities** (Broken weave lines)

## Key Features
* **Dual Classical & Logic Pipeline:** Adaptive background subtraction, Sauvola thresholding, multi-scale FFT (512/256/128), and DWT (`db4`) detail analysis
* **Precise Polygon Contour Masks:** Sub-pixel Douglas-Peucker polygon approximation and alpha-blended defect segmentation
* **Per-Defect Geometry & Photometrics:** Exact Area, Perimeter, Circularity, Aspect Ratio, Major Axis length, and local Euclidean color deltas ($\Delta\text{RGB}$, $\Delta\text{HSV}$)
* **Optional Optical Calibration:** Millimeter-per-pixel (`mm_per_pixel`) scaling with automatic conversion between pixel and physical metrics ($\text{mm}^2$, inches)
* **Interactive UI:** Streamlit dashboard with overlay style toggles (Polygons / Boxes / Both), before/after comparison slider, and diagnostic frequency saliency maps
* **Batch Processing & Heatmaps:** Multi-image quality reports with defect spatial density clustering
* **Automated Test Suite & Benchmarking:** Pytest verification and standardized AITEX dataset evaluation harness

## Architecture

```
FabricQA/
├── app.py                      # Streamlit dashboard (UI + inspection modes + heatmaps)
├── config.py                   # Pipeline settings & physical calibration parameters
├── requirements.txt            # Project dependencies
├── pytest.ini                  # Pytest configuration
├── inspectors/
│   ├── __init__.py             # Package exports (unified_processor, TextureInspector)
│   ├── unified_processor.py    # Orchestrator running parallel CV pipelines
│   ├── geometry.py             # Polygon approximation & per-defect geometric/photometric metrics
│   ├── classic_edge.py         # Adaptive background subtraction + CLAHE + Sauvola (Hole, Tear, Oil Stain, Puckering)
│   ├── logic_spectral.py       # Multi-resolution FFT (512/256/128) + DWT db4 (Slub, Skip/Miss Stitch)
│   ├── logic_seam.py           # Hough deskew + thread mask + 1D projection profiling (Seam defects)
│   └── texture_inspector.py    # 18-filter Gabor bank + sliding GLCM Haralick + LBP/Entropy (Weave Irregularity, Snag/Knot)
├── eval/
│   ├── run_eval.py             # Evaluation harness (AUROC, F1, FPR, IoU, 1024x1024 latency benchmark)
│   ├── run_ablation.py         # Phase 2 Texture feature bank ablation suite
│   ├── baseline.md             # Baseline evaluation metrics on AITEX dataset
│   ├── phase1_eval.md          # Phase 1 evaluation metrics after measurement foundation
│   ├── phase2_ablation.md      # Phase 2 ablation study results (LBP vs Gabor vs GLCM vs Combined)
│   └── baseline_results.csv    # Per-sample predictions and metrics
└── tests/
    ├── test_inspectors.py      # Contract & regression tests for inspectors
    ├── test_geometry.py        # Polygon, geometric & photometric unit tests
    ├── test_texture.py         # Gabor caching, GLCM levels, and LBP unit tests
    └── test_eval_harness.py    # Evaluation harness unit tests
```

## How It Works

| Inspector | Algorithm | Detects | Status |
|-----------|-----------|---------|:---:|
| **Classic Edge** (`classic_edge.py`) | Adaptive background subtraction + Sauvola dynamic thresholding + morphology | Holes, Tears, Oil Stains, Puckering | Implemented & Wired |
| **Logic Spectral** (`logic_spectral.py`) | Multi-scale FFT Spectral Residual (512, 256, 128) + DWT (`db4`) detail decomposition | Slub, Skip/Miss Stitch | Implemented & Wired |
| **Logic Seam** (`logic_seam.py`) | Hough transform deskewing + stitch segmentation + 1D projection profile + 1D FFT periodicity | Skip Stitch, Broken Stitch, Run-off Stitch, Crooked Stitch | Implemented & Wired |
| **Texture Inspector** (`texture_inspector.py`) | 18-filter real Gabor bank (6 orientations x 3 scales) + Sliding GLCM (contrast, correlation, energy, homogeneity) + LBP uniform & Shannon entropy | Weave Irregularity, Snag / Knot | Implemented & Wired |
| **Unified Processor** (`unified_processor.py`) | Multi-pipeline orchestration, defect deduplication, taxonomy classification | All structural and surface categories | Implemented & Wired |

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/NareshKumar241205/Fabrict-Defect-Detection.git
    cd Fabrict-Defect-Detection
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    .venv\Scripts\activate    # Windows
    # source .venv/bin/activate  # Linux/Mac
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the application using Streamlit:

```bash
streamlit run app.py
```

### Running Tests

```bash
pytest tests/ -v
```

## Inspection Modes

| Mode | Description |
|------|-------------|
| 🔬 **Full Inspection** | Runs all inspectors (Texture + Spectral + Seam + Edge) |
| 🧵 **Texture Analysis** | LBP + Entropy + Gabor directional filters |
| 📡 **Spectral (FFT)** | Multi-resolution frequency saliency |
| 🪡 **Seam / Stitch** | Skip-stitch and thread gap detection |
| 📐 **Edge / Structure** | Wrinkle, fold, and weave irregularity detection |
|  **Batch Inspection** | Process multiple images with combined report |
| 📊 **History** | View past inspection results and defect trends |

## License

This project is open source and available under the [MIT License](LICENSE).

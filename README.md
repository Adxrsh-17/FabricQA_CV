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
* **5 Inspection Modes:** Full, Texture, Spectral, Seam, Edge/Structural
* **Batch Processing:** Upload multiple images and get a combined CSV report
* **Inspection History:** Track quality trends over time with charts
* **Gabor Filter Bank:** 18 directional filters to catch diagonal/oriented defects
* **Multi-Resolution FFT:** Spectral analysis at multiple scales for micro & macro defects
* **Smart Confidence Scores:** Z-score based confidence metrics (not hardcoded)
* **Auto-Calibration:** Statistical thresholding adapts to each image
* **Live Camera:** Real-time webcam inspection
* **Automated Reporting:** Downloadable CSV defect reports

## Architecture

```
FabricQA/
├── app.py                      # Streamlit dashboard (UI + inspection modes + heatmaps)
├── config.py                   # Pipeline settings (Classic, Spectral, Seam, Texture)
├── requirements.txt            # Project dependencies
├── pytest.ini                  # Pytest configuration
├── inspectors/
│   ├── __init__.py             # Package exports (unified_processor)
│   ├── unified_processor.py    # Orchestrator running parallel CV pipelines
│   ├── classic_edge.py         # Adaptive background subtraction + CLAHE + Sauvola (Hole, Tear, Oil Stain)
│   ├── logic_spectral.py       # Multi-resolution FFT (512/256/128) + DWT db4 (Slub, Skip/Miss Stitch)
│   └── logic_seam.py           # Hough deskew + thread mask + 1D projection profiling (Seam defects)
├── eval/
│   ├── run_eval.py             # Evaluation harness (AUROC, F1, FPR, IoU, 1024x1024 latency benchmark)
│   ├── baseline.md             # Baseline evaluation metrics on AITEX dataset
│   └── baseline_results.csv    # Per-sample predictions and metrics
└── tests/
    ├── test_inspectors.py      # Contract & regression tests for inspectors
    └── test_eval_harness.py    # Evaluation harness unit tests
```

## How It Works

| Inspector | Algorithm | Detects | Status |
|-----------|-----------|---------|:---:|
| **Classic Edge** (`classic_edge.py`) | Adaptive background subtraction + Sauvola dynamic thresholding + morphology | Holes, Tears, Oil Stains, Puckering | Implemented & Wired |
| **Logic Spectral** (`logic_spectral.py`) | Multi-scale FFT Spectral Residual (512, 256, 128) + DWT (`db4`) detail decomposition | Slub, Skip/Miss Stitch | Implemented & Wired |
| **Logic Seam** (`logic_seam.py`) | Hough transform deskewing + stitch segmentation + 1D projection profile + 1D FFT periodicity | Skip Stitch, Broken Stitch, Run-off Stitch, Crooked Stitch | Implemented & Wired |
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

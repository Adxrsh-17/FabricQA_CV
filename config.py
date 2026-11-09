# config.py
import numpy as np
from typing import Optional

# ==========================================
# CALIBRATION & MEASUREMENT SETTINGS
# ==========================================
CALIBRATION_SETTINGS = {
    # Physical scale: millimeters per pixel.
    # Set to None if uncalibrated (pixel units used, physical features disabled).
    "MM_PER_PIXEL": None,
    # Alternative calibration via known reference object
    "KNOWN_REFERENCE_LENGTH_MM": None,
    "KNOWN_REFERENCE_PIXELS": None,
}

def get_mm_per_pixel(
    mm_per_pixel_override: Optional[float] = None,
    ref_length_mm: Optional[float] = None,
    ref_pixels: Optional[float] = None,
) -> Optional[float]:
    """Resolve mm_per_pixel from explicit override, reference pair, or config."""
    if mm_per_pixel_override is not None and mm_per_pixel_override > 0:
        return float(mm_per_pixel_override)

    if ref_length_mm is not None and ref_pixels is not None and ref_pixels > 0:
        return float(ref_length_mm) / float(ref_pixels)

    cfg_mm = CALIBRATION_SETTINGS.get("MM_PER_PIXEL")
    if cfg_mm is not None and cfg_mm > 0:
        return float(cfg_mm)

    cfg_len = CALIBRATION_SETTINGS.get("KNOWN_REFERENCE_LENGTH_MM")
    cfg_px = CALIBRATION_SETTINGS.get("KNOWN_REFERENCE_PIXELS")
    if cfg_len is not None and cfg_px is not None and cfg_px > 0:
        return float(cfg_len) / float(cfg_px)

    return None

# ==========================================
# GEOMETRY & CONTOUR SETTINGS
# ==========================================
GEOMETRY_SETTINGS = {
    "APPROX_EPSILON_FACTOR": 0.015,
    "MIN_CONTOUR_POINTS": 3,
    "OVERLAY_ALPHA": 0.35,
    "CONTOUR_THICKNESS": 2,
    "BORDER_MARGIN_PX": 8,  # Suppress edge artifacts near image boundaries
}

# ==========================================
# PIPELINE 1: CLASSICAL CV (Oil, Hole, Tear)
# ==========================================
CLASSIC_SETTINGS = {
    "IMAGE_RESIZE_WIDTH": 800,
    "HOLE_AREA_MIN": 800,
    "MAX_BOX_AREA_RATIO": 0.22,
    "TEAR_ASPECT_RATIO_MIN": 2.0,
    "PUCKER_DARKNESS_RATIO": 0.70,
    "PUCKER_TEXTURE_RATIO_MAX": 0.25,
}

# ==========================================
# PIPELINE 2: LOGIC CHANGE (Slub, Skip/Miss/Crooked Stitch)
# ==========================================
LOGIC_SPECTRAL_SETTINGS = {
    "IMAGE_RESIZE_WIDTH": 800,
    "SMOOTHING_KERNEL": 5,
    "USE_DWT": True,
    "BOUNDARY_MARGIN_RATIO": 0.02,  # Filter out FFT boundary wrap-around artifacts
}

LOGIC_SEAM_SETTINGS = {
    "STITCH_COLOR_THRESH": 180,
    "GAP_TOLERANCE": 15,
    "CROOKED_R2_THRESH": 0.85,
    "RUNOFF_EDGE_MARGIN": 0.10,
    "BROKEN_GAP_MIN": 20,
}

# ==========================================
# PIPELINE 3: SEAM INSPECTOR
# ==========================================
SEAM_SETTINGS = {
    "STITCH_COLOR_THRESH": 180,
    "GAP_TOLERANCE": 10,
    "CROOKED_R2_THRESH": 0.85,
    "PUCKER_VAR_SIGMA": 3.0,
    "RUNOFF_EDGE_MARGIN": 0.10,
    "BROKEN_GAP_MIN": 20,
}

# ==========================================
# PIPELINE 4: TEXTURE INSPECTOR (Gabor + GLCM + LBP/Entropy)
# ==========================================
TEXTURE_SETTINGS = {
    "IMAGE_RESIZE_WIDTH": 800,
    "MIN_DEFECT_AREA": 300,
    "FEATURE_MODE": "all",  # "lbp_entropy", "gabor", "glcm", "all"
    "FUSION_METHOD": "max",  # "max" or "weighted_sum"
    # Gabor Filter Bank: 6 orientations x 3 scales = 18 filters
    "GABOR_ORIENTATIONS": [0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6],
    "GABOR_SCALES": [4.0, 8.0, 16.0],  # Wavelengths (lambda)
    "GABOR_KSIZE": 21,
    "GABOR_GAMMA": 0.5,
    "GABOR_PSI": 0.0,
    # LBP + Local Entropy
    "LBP_POINTS": 8,
    "LBP_RADIUS": 1,
    "ENTROPY_DISK_RADIUS": 5,
}

# ==========================================
# GLCM SETTINGS (for texture inspector)
# ==========================================
GLCM_SETTINGS = {
    "WINDOW_SIZE": 32,
    "STEP_SIZE": 16,
    "NUM_LEVELS": 32,  # Configurable gray levels: 16, 32, 64
    "DISTANCES": [1, 2],
    "ANGLES": [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
    "FEATURES": ["contrast", "correlation", "energy", "homogeneity"],
    "THRESHOLDS": {
        "contrast_max": 250,
        "correlation_min": 0.80,
        "homogeneity_max": 0.98,
    },
}

# ==========================================
# UNIFIED SETTINGS (shared across inspectors)
# ==========================================
UNIFIED_SETTINGS = {
    "USE_SAUVOLA_TEXTURE": True,
    "SAUVOLA_WINDOW": 151,
}

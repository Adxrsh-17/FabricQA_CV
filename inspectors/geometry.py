"""inspectors/geometry.py
Per-defect geometric and photometric feature extraction.

Computes:
- Contour smoothing via morphological operations and cv2.approxPolyDP
- Precise polygon coordinates (mapped to original image scale)
- Geometric descriptors: Area, Perimeter, Circularity, Aspect Ratio, Major Axis Length
- Photometric descriptors: Mean RGB/HSV delta vs. surrounding background
- Physical unit conversion (if mm_per_pixel calibration is provided)
"""

from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np


def smooth_and_approx_contour(
    contour: np.ndarray,
    epsilon_factor: float = 0.015,
) -> np.ndarray:
    """Smooth contour and approximate with polygon using Douglas-Peucker algorithm."""
    if contour is None or len(contour) == 0:
        return np.array([], dtype=np.int32)
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True)
    return approx


def compute_defect_photometrics(
    img_bgr: np.ndarray,
    img_hsv: np.ndarray,
    contour: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> Dict[str, float]:
    """Calculate mean RGB and HSV Euclidean color delta between defect interior

    and local surrounding background annulus.
    """
    x, y, w, h = bbox
    img_h, img_w = img_bgr.shape[:2]

    # Create masks for defect interior and local neighborhood
    defect_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.drawContours(defect_mask, [contour], -1, 255, -1)

    # Neighborhood ring (dilate defect by proportional margin)
    expand = max(8, int(min(w, h) * 0.4))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand * 2 + 1, expand * 2 + 1))
    dilated_mask = cv2.dilate(defect_mask, kernel)
    surround_mask = cv2.subtract(dilated_mask, defect_mask)

    defect_pixels_bgr = img_bgr[defect_mask > 0]
    surround_pixels_bgr = img_bgr[surround_mask > 0]

    defect_pixels_hsv = img_hsv[defect_mask > 0]
    surround_pixels_hsv = img_hsv[surround_mask > 0]

    if defect_pixels_bgr.size > 0 and surround_pixels_bgr.size > 0:
        mean_defect_bgr = np.mean(defect_pixels_bgr, axis=0)
        mean_surround_bgr = np.mean(surround_pixels_bgr, axis=0)
        delta_rgb = float(np.linalg.norm(mean_defect_bgr - mean_surround_bgr))

        mean_defect_hsv = np.mean(defect_pixels_hsv, axis=0)
        mean_surround_hsv = np.mean(surround_pixels_hsv, axis=0)
        delta_hsv = float(np.linalg.norm(mean_defect_hsv - mean_surround_hsv))
    else:
        delta_rgb = 0.0
        delta_hsv = 0.0

    return {
        "delta_rgb": round(delta_rgb, 2),
        "delta_hsv": round(delta_hsv, 2),
    }


def extract_defect_geometry(
    contour: np.ndarray,
    img_bgr: np.ndarray,
    img_hsv: np.ndarray,
    scale: float,
    mm_per_pixel: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute complete geometric and photometric attributes for a detected defect contour."""
    # 1. Smooth and approximate polygon
    approx = smooth_and_approx_contour(contour, epsilon_factor=0.015)
    if len(approx) < 3:
        approx = contour

    # Map polygon coordinates to original image scale
    orig_polygon = (approx.astype(np.float32) / scale).astype(np.int32).reshape(-1, 2).tolist()

    # 2. Bounding Box
    x, y, w, h = cv2.boundingRect(approx)
    ox = max(0, int(x / scale))
    oy = max(0, int(y / scale))
    ow = max(1, int(w / scale))
    oh = max(1, int(h / scale))

    # 3. Geometric features (at original scale)
    proc_area = float(cv2.contourArea(approx))
    real_area_px = max(1, int(proc_area / (scale ** 2)))

    proc_peri = float(cv2.arcLength(approx, True))
    real_peri_px = max(1.0, proc_peri / scale)

    # Circularity (isoperimetric quotient: 4 * pi * area / peri^2, in [0, 1])
    circularity = min(1.0, max(0.0, (4.0 * np.pi * proc_area) / max(proc_peri ** 2, 1e-6)))

    aspect_ratio = float(w) / max(float(h), 1.0)

    # Major and minor axis length via minimum area rotated bounding box
    if len(approx) >= 3:
        rect = cv2.minAreaRect(approx)
        dim1, dim2 = rect[1]
        major_axis_px = max(dim1, dim2) / scale
        minor_axis_px = min(dim1, dim2) / scale
    else:
        major_axis_px = max(ow, oh)
        minor_axis_px = min(ow, oh)

    # 4. Photometric color deltas
    color_metrics = compute_defect_photometrics(img_bgr, img_hsv, approx, (x, y, w, h))

    # 5. Physical units conversion if calibrated
    physical_metrics = {}
    if mm_per_pixel is not None and mm_per_pixel > 0:
        area_mm2 = real_area_px * (mm_per_pixel ** 2)
        major_axis_mm = major_axis_px * mm_per_pixel
        major_axis_in = major_axis_mm / 25.4
        physical_metrics = {
            "area_mm2": round(area_mm2, 3),
            "major_axis_mm": round(major_axis_mm, 2),
            "major_axis_in": round(major_axis_in, 3),
            "calibrated": True,
        }
    else:
        physical_metrics = {
            "area_mm2": None,
            "major_axis_mm": None,
            "major_axis_in": None,
            "calibrated": False,
        }

    return {
        "bbox_x": ox,
        "bbox_y": oy,
        "bbox_w": ow,
        "bbox_h": oh,
        "polygon": orig_polygon,
        "area_px": real_area_px,
        "perimeter_px": round(real_peri_px, 1),
        "circularity": round(circularity, 3),
        "aspect_ratio": round(aspect_ratio, 2),
        "major_axis_px": round(major_axis_px, 1),
        "minor_axis_px": round(minor_axis_px, 1),
        "delta_rgb": color_metrics["delta_rgb"],
        "delta_hsv": color_metrics["delta_hsv"],
        **physical_metrics,
    }

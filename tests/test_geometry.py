import io
import cv2
import numpy as np
import pytest

from inspectors.geometry import (
    smooth_and_approx_contour,
    compute_defect_photometrics,
    extract_defect_geometry,
)
from inspectors.unified_processor import unified_processor


def test_smooth_and_approx_contour():
    # Create jittery rectangle contour
    pts = np.array([
        [10, 10], [50, 11], [100, 10],
        [101, 50], [100, 100],
        [50, 99], [10, 100],
        [9, 50]
    ], dtype=np.int32).reshape((-1, 1, 2))

    approx = smooth_and_approx_contour(pts, epsilon_factor=0.02)
    assert len(approx) <= len(pts)
    assert len(approx) >= 4


def test_compute_defect_photometrics():
    img_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    # Bright red square in center
    img_bgr[30:70, 30:70] = (0, 0, 255)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    contour = np.array([
        [30, 30], [70, 30], [70, 70], [30, 70]
    ], dtype=np.int32).reshape((-1, 1, 2))

    metrics = compute_defect_photometrics(img_bgr, img_hsv, contour, (30, 30, 40, 40))
    assert metrics["delta_rgb"] > 50.0
    assert metrics["delta_hsv"] > 50.0


def test_extract_defect_geometry_uncalibrated():
    img_bgr = np.full((100, 100, 3), 128, dtype=np.uint8)
    cv2.circle(img_bgr, (50, 50), 20, (20, 20, 20), -1)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Circle contour
    contour, _ = cv2.findContours((img_bgr[:, :, 0] < 50).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contour) > 0

    geom = extract_defect_geometry(contour[0], img_bgr, img_hsv, scale=1.0, mm_per_pixel=None)
    assert geom["calibrated"] is False
    assert geom["area_mm2"] is None
    assert geom["area_px"] > 1000
    assert 0.7 <= geom["circularity"] <= 1.0  # Circles have circularity near 1.0
    assert isinstance(geom["polygon"], list)
    assert len(geom["polygon"]) >= 3
    assert len(geom["polygon"][0]) == 2


def test_extract_defect_geometry_calibrated():
    img_bgr = np.full((100, 100, 3), 128, dtype=np.uint8)
    cv2.circle(img_bgr, (50, 50), 20, (20, 20, 20), -1)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    contour, _ = cv2.findContours((img_bgr[:, :, 0] < 50).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mm_scale = 0.05  # 0.05 mm/pixel
    geom = extract_defect_geometry(contour[0], img_bgr, img_hsv, scale=1.0, mm_per_pixel=mm_scale)
    assert geom["calibrated"] is True
    assert geom["area_mm2"] is not None
    assert abs(geom["area_mm2"] - (geom["area_px"] * (mm_scale ** 2))) < 1e-2
    assert geom["major_axis_mm"] is not None
    assert geom["major_axis_in"] is not None


def test_unified_processor_passes_polygon_and_calibration():
    img = np.full((400, 400, 3), 150, dtype=np.uint8)
    # Add hole defect
    cv2.circle(img, (200, 200), 30, (20, 20, 20), -1)
    _, enc = cv2.imencode(".png", img)
    buf = io.BytesIO(enc.tobytes())

    res = unified_processor.process(buf, sensitivity=2.0, mm_per_pixel=0.08)
    assert "defects" in res
    assert len(res["defects"]) > 0
    d0 = res["defects"][0]
    assert "polygon" in d0
    assert isinstance(d0["polygon"], list)
    assert "Perimeter (px)" in d0
    assert "Circularity" in d0
    assert "Delta RGB" in d0
    assert d0["Calibrated"] is True
    assert d0["Area (mm2)"] is not None

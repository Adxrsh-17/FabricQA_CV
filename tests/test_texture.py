import io
import time
import cv2
import numpy as np
import pytest

from inspectors.texture_inspector import TextureInspector


def _create_synthetic_fabric_buffer(size=(300, 300), with_slub=True) -> io.BytesIO:
    """Generate in-memory synthetic fabric image with weave lines and optional slub defect."""
    h, w = size
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)
    base = 128 + 30 * np.sin(2 * np.pi * xx / 6.0) + 30 * np.cos(2 * np.pi * yy / 6.0)
    noise = np.random.normal(0, 4, (h, w))
    img = np.clip(base + noise, 0, 255).astype(np.uint8)

    if with_slub:
        # Thick horizontal yarn slub defect
        img[h // 2 - 4: h // 2 + 4, w // 4: 3 * w // 4] = 240

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    _, enc = cv2.imencode(".png", img_bgr)
    return io.BytesIO(enc.tobytes())


def test_gabor_bank_initialization_and_caching():
    inspector = TextureInspector()
    assert len(inspector._gabor_kernels) == 18
    for item in inspector._gabor_kernels:
        assert "kernel" in item
        assert "theta" in item
        assert "lambd" in item
        assert item["kernel"].shape == (21, 21)


def test_glcm_levels_and_saliency():
    inspector = TextureInspector()
    img_gray = np.full((100, 100), 128, dtype=np.uint8)
    # Add an anomaly block
    img_gray[30:60, 30:60] = 20

    for levels in [16, 32, 64]:
        glcm_map = inspector._compute_glcm_saliency(img_gray, num_levels=levels, window_size=32, step=16)
        assert glcm_map.shape == (100, 100)
        assert glcm_map.dtype == np.uint8
        assert np.max(glcm_map) > 0


def test_lbp_entropy_saliency():
    inspector = TextureInspector()
    img_gray = np.full((120, 120), 128, dtype=np.uint8)
    img_gray[40:80, 40:80] = np.random.randint(0, 255, (40, 40), dtype=np.uint8)

    lbp_ent = inspector._compute_lbp_entropy_saliency(img_gray)
    assert lbp_ent.shape == (120, 120)
    assert lbp_ent.dtype == np.uint8


def test_texture_inspector_contract_and_modes():
    inspector = TextureInspector()
    buf = _create_synthetic_fabric_buffer(size=(300, 300), with_slub=True)

    for mode in ["lbp_entropy", "gabor", "glcm", "all"]:
        buf.seek(0)
        defects, viz_maps = inspector.detect_defects(buf, sensitivity=2.0, feature_mode=mode)
        assert isinstance(defects, list)
        assert isinstance(viz_maps, dict)
        assert "texture_saliency_map" in viz_maps
        assert "grayscale_clahe" in viz_maps

        if mode in ["gabor", "all"]:
            assert "gabor_saliency_map" in viz_maps
        if mode in ["glcm", "all"]:
            assert "glcm_anomaly_map" in viz_maps

        for d in defects:
            assert "Type" in d
            assert "Quality Score" in d
            assert "polygon" in d
            assert "bbox_x" in d
            assert "Perimeter (px)" in d
            assert "Circularity" in d


def test_gabor_runtime_1024():
    inspector = TextureInspector()
    img_gray = np.random.randint(0, 255, (1024, 1024), dtype=np.uint8)

    t0 = time.perf_counter()
    gabor_map = inspector._compute_gabor_saliency(img_gray)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert gabor_map.shape == (1024, 1024)
    # 18 Gabor convolutions on 1024x1024 should complete well within 1.5 seconds on modern CPU
    assert elapsed_ms < 1500.0


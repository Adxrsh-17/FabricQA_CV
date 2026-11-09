import os
import numpy as np
import pytest
from eval.run_eval import (
    load_and_orient_strip,
    evaluate_single_strip,
    benchmark_1024x1024,
    UnifiedProcessor,
)


def test_load_and_orient_strip(tmp_path):
    import cv2
    # Create vertical strip 4096x256 to test rotation
    vertical_img = np.zeros((4096, 256, 3), dtype=np.uint8)
    img_path = str(tmp_path / "sample.png")
    cv2.imwrite(img_path, vertical_img)

    mask_img = np.zeros((4096, 256), dtype=np.uint8)
    mask_img[100:200, 50:100] = 255
    mask_path = str(tmp_path / "sample_mask.png")
    cv2.imwrite(mask_path, mask_img)

    img, mask = load_and_orient_strip(img_path, [mask_path])
    assert img is not None
    assert mask is not None
    # Must be rotated to horizontal orientation (height=256, width=4096)
    assert img.shape[:2] == (256, 4096)
    assert mask.shape[:2] == (256, 4096)


def test_evaluate_single_strip(tmp_path):
    import cv2
    img = np.full((256, 1024, 3), 128, dtype=np.uint8)
    # Draw a defect
    cv2.circle(img, (512, 128), 25, (20, 20, 20), -1)
    img_path = str(tmp_path / "strip.png")
    cv2.imwrite(img_path, img)

    mask = np.zeros((256, 1024), dtype=np.uint8)
    cv2.circle(mask, (512, 128), 25, 255, -1)
    mask_path = str(tmp_path / "strip_mask.png")
    cv2.imwrite(mask_path, mask)

    strip_info = {
        "id": "test_strip_01",
        "img_path": img_path,
        "mask_paths": [mask_path],
        "ground_truth": 1,
        "category": "Defective",
    }
    processor = UnifiedProcessor()
    res = evaluate_single_strip(strip_info, processor, tile_w=1024)

    assert "ground_truth" in res
    assert "pred_detected" in res
    assert "latency_ms" in res
    assert res["ground_truth"] == 1


def test_benchmark_1024x1024():
    bench = benchmark_1024x1024(runs=2)
    assert bench["mean_ms"] > 0
    assert bench["min_ms"] > 0

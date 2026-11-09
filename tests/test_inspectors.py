import io
import cv2
import numpy as np
import pytest

from inspectors.classic_edge import ClassicEdgeInspector
from inspectors.logic_spectral import LogicSpectralInspector
from inspectors.logic_seam import LogicSeamInspector
from inspectors.unified_processor import UnifiedProcessor, unified_processor


def _create_test_image_buffer(color=(128, 128, 128), size=(400, 400), with_defect=False):
    """Helper to generate an in-memory encoded JPEG/PNG buffer."""
    img = np.full((size[0], size[1], 3), color, dtype=np.uint8)
    if with_defect:
        # Draw a synthetic dark spot (hole/stain)
        cv2.circle(img, (size[1] // 2, size[0] // 2), 25, (20, 20, 20), -1)
    success, encoded = cv2.imencode(".png", img)
    assert success
    return io.BytesIO(encoded.tobytes())


def test_classic_edge_contract():
    inspector = ClassicEdgeInspector()
    buf = _create_test_image_buffer(with_defect=True)
    defects, viz_maps = inspector.detect_defects(buf)

    assert isinstance(defects, list)
    assert isinstance(viz_maps, dict)
    for d in defects:
        for key in ["Type", "Area (px)", "Quality Score", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "Pipeline", "Category"]:
            assert key in d


def test_logic_spectral_contract():
    inspector = LogicSpectralInspector()
    buf = _create_test_image_buffer(with_defect=True)
    defects, viz_maps = inspector.detect_defects(buf)

    assert isinstance(defects, list)
    assert isinstance(viz_maps, dict)
    assert "logic_spectral_map" in viz_maps


def test_logic_seam_regression_out_of_bounds_density():
    """Regression test: when stitch density is outside [0.015, 0.35],

    detect_defects must return a dictionary for viz_maps, NOT an ndarray.
    """
    inspector = LogicSeamInspector()
    # Plain solid gray image has 0 stitch density -> triggers density check
    buf = _create_test_image_buffer(color=(200, 200, 200), size=(300, 600), with_defect=False)
    defects, viz_maps = inspector.detect_defects(buf)

    assert isinstance(defects, list)
    assert isinstance(viz_maps, dict), f"Expected dict for viz_maps, got {type(viz_maps)}"
    assert "logic_seam_map" in viz_maps

    # Test with solid black (dark stitch check)
    buf_black = _create_test_image_buffer(color=(0, 0, 0), size=(300, 600), with_defect=False)
    defects_b, viz_maps_b = inspector.detect_defects(buf_black)
    assert isinstance(defects_b, list)
    assert isinstance(viz_maps_b, dict)


def test_unified_processor_contract():
    processor = UnifiedProcessor()
    buf = _create_test_image_buffer(with_defect=True)
    result = processor.process(buf)

    assert isinstance(result, dict)
    assert "defects" in result
    assert "viz_maps" in result
    assert "summary" in result
    assert isinstance(result["defects"], list)
    assert isinstance(result["viz_maps"], dict)


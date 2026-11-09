"""inspectors/texture_inspector.py
PIPELINE 4: Texture Inspection Engine using Gabor Filter Bank, GLCM Haralick Features,
and Local Binary Patterns (LBP) + Local Entropy.

Reference-free anomaly detection using image-level z-score statistics.
"""

import time
import logging
from typing import Tuple, List, Dict, Any, BinaryIO, Optional
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk

from config import TEXTURE_SETTINGS, GLCM_SETTINGS, GEOMETRY_SETTINGS
from inspectors.geometry import extract_defect_geometry

logger = logging.getLogger(__name__)


class TextureInspector:
    """Detects texture irregularities, slubs, and weave disruptions via Gabor filter banks,

    GLCM Haralick descriptors, and LBP + Local Entropy.
    """

    def __init__(self):
        self.defects: List[Dict[str, Any]] = []
        # Precompute and cache the 18 real Gabor kernels (6 orientations x 3 scales)
        self._gabor_kernels = self._precompute_gabor_kernels()

    def _precompute_gabor_kernels(self) -> List[Dict[str, Any]]:
        """Precompute and cache 18 real Gabor filter kernels for real-time performance."""
        kernels = []
        orientations = TEXTURE_SETTINGS.get("GABOR_ORIENTATIONS", [
            0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6
        ])
        scales = TEXTURE_SETTINGS.get("GABOR_SCALES", [4.0, 8.0, 16.0])
        ksize = TEXTURE_SETTINGS.get("GABOR_KSIZE", 21)
        gamma = TEXTURE_SETTINGS.get("GABOR_GAMMA", 0.5)
        psi = TEXTURE_SETTINGS.get("GABOR_PSI", 0.0)

        for theta in orientations:
            for lambd in scales:
                sigma = 0.56 * lambd  # Standard 1-octave bandwidth relationship
                kernel = cv2.getGaborKernel(
                    (ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F
                )
                kernels.append({
                    "kernel": kernel,
                    "theta": theta,
                    "lambd": lambd,
                })
        return kernels

    def _preprocess(self, img_buffer: BinaryIO) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        if hasattr(img_buffer, "seek"):
            img_buffer.seek(0)
        file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        h, w = img.shape[:2]
        target_w = TEXTURE_SETTINGS.get("IMAGE_RESIZE_WIDTH", 800)
        scale = target_w / w
        target_h = int(h * scale)

        img_small = cv2.resize(img, (target_w, target_h))
        img_gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_gray = clahe.apply(img_gray)

        return img, img_small, img_gray, scale

    def _compute_gabor_saliency(self, img_gray: np.ndarray) -> np.ndarray:
        """Filter image through the 18-filter Gabor bank, normalize responses by

        their defect-free statistics, and fuse channels via max-pooling.
        """
        img_f = img_gray.astype(np.float32)
        norm_responses = []

        for item in self._gabor_kernels:
            k = item["kernel"]
            filtered = cv2.filter2D(img_f, cv2.CV_32F, k)
            energy = np.abs(filtered)

            # Normalization by channel statistics (z-score deviation)
            med = float(np.median(energy))
            # Robust standard deviation via MAD (Median Absolute Deviation)
            mad = float(np.median(np.abs(energy - med)))
            sigma = max(1.4826 * mad, 1e-4)

            z_map = np.abs(energy - med) / sigma
            norm_responses.append(z_map)

        # Channel fusion via max-pooling across all 18 orientations and scales
        fused = np.maximum.reduce(norm_responses)
        saliency = cv2.normalize(fused, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return saliency

    def _compute_glcm_saliency(
        self,
        img_gray: np.ndarray,
        num_levels: int = 32,
        window_size: int = 32,
        step: int = 16,
    ) -> np.ndarray:
        """Compute the 4 most discriminative Haralick features (contrast, correlation,

        energy, homogeneity) over sliding windows, scoring each window by its
        z-score deviation from the image's own global statistics.
        """
        h, w = img_gray.shape[:2]
        # Quantize to configurable gray levels (16, 32, or 64)
        div = max(1, 256 // num_levels)
        img_quant = np.clip(img_gray // div, 0, num_levels - 1).astype(np.uint8)

        distances = GLCM_SETTINGS.get("DISTANCES", [1, 2])
        angles = GLCM_SETTINGS.get("ANGLES", [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])

        y_steps = list(range(0, max(1, h - window_size + 1), step))
        x_steps = list(range(0, max(1, w - window_size + 1), step))
        if not y_steps:
            y_steps = [0]
        if not x_steps:
            x_steps = [0]

        feat_records = []
        grid_coords = []

        for y in y_steps:
            for x in x_steps:
                patch = img_quant[y:y + window_size, x:x + window_size]
                if patch.shape[0] < 4 or patch.shape[1] < 4:
                    continue

                glcm = graycomatrix(
                    patch,
                    distances=distances,
                    angles=angles,
                    levels=num_levels,
                    symmetric=True,
                    normed=True,
                )

                contrast = float(np.mean(graycoprops(glcm, "contrast")))
                correlation = float(np.mean(graycoprops(glcm, "correlation")))
                energy = float(np.mean(graycoprops(glcm, "energy")))
                homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))

                feat_records.append([contrast, correlation, energy, homogeneity])
                grid_coords.append((y, x))

        if not feat_records:
            return np.zeros((h, w), dtype=np.uint8)

        feat_arr = np.array(feat_records, dtype=np.float32)  # Shape (N, 4)
        means = np.mean(feat_arr, axis=0)
        stds = np.std(feat_arr, axis=0)
        stds[stds < 1e-5] = 1.0

        # Multi-feature z-score anomaly score per window
        z_scores = np.abs(feat_arr - means) / stds
        window_anomaly = np.mean(z_scores, axis=1)

        # Assemble into grid map
        grid_h = len(y_steps)
        grid_w = len(x_steps)
        grid_map = np.zeros((grid_h, grid_w), dtype=np.float32)

        idx = 0
        for i in range(grid_h):
            for j in range(grid_w):
                if idx < len(window_anomaly):
                    grid_map[i, j] = window_anomaly[idx]
                    idx += 1

        # Smooth and upsample back to original image size
        grid_smoothed = cv2.GaussianBlur(grid_map, (3, 3), 0)
        upsampled = cv2.resize(grid_smoothed, (w, h), interpolation=cv2.INTER_LINEAR)
        glcm_saliency = cv2.normalize(upsampled, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return glcm_saliency

    def _compute_lbp_entropy_saliency(self, img_gray: np.ndarray) -> np.ndarray:
        """Compute Local Binary Patterns (uniform) and local entropy texture saliency."""
        # 1. LBP
        P = TEXTURE_SETTINGS.get("LBP_POINTS", 8)
        R = TEXTURE_SETTINGS.get("LBP_RADIUS", 1)
        lbp = local_binary_pattern(img_gray, P=P, R=R, method="uniform")

        # Texture irregularity via local variance of LBP codes
        lbp_blur = cv2.GaussianBlur(lbp.astype(np.float32), (9, 9), 0)
        lbp_diff = np.abs(lbp.astype(np.float32) - lbp_blur)
        lbp_sal = cv2.normalize(lbp_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 2. Local Entropy (sampled grid for high speed)
        disk_rad = TEXTURE_SETTINGS.get("ENTROPY_DISK_RADIUS", 5)
        small_gray = cv2.resize(img_gray, (img_gray.shape[1] // 2, img_gray.shape[0] // 2))
        ent = rank_entropy(small_gray, disk(disk_rad))
        ent_full = cv2.resize(ent, (img_gray.shape[1], img_gray.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Mean z-score anomaly
        ent_f = ent_full.astype(np.float32)
        ent_med = np.median(ent_f)
        ent_std = max(float(np.std(ent_f)), 1e-4)
        ent_z = np.clip((np.abs(ent_f - ent_med) / ent_std) * 40, 0, 255).astype(np.uint8)

        # Fuse LBP + Entropy
        lbp_entropy_map = cv2.addWeighted(lbp_sal, 0.5, ent_z, 0.5, 0)
        return lbp_entropy_map

    def detect_defects(
        self,
        img_buffer: BinaryIO,
        sensitivity: float = 2.5,
        mm_per_pixel: Optional[float] = None,
        feature_mode: str = "all",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
        """Detect texture anomalies using Gabor, GLCM, or fused feature bank.

        Args:
            img_buffer: Input fabric image buffer
            sensitivity: Detection threshold multiplier (z-score multiplier)
            mm_per_pixel: Physical calibration scale (mm/px)
            feature_mode: "lbp_entropy", "gabor", "glcm", or "all"
        """
        img, img_small, img_gray, scale = self._preprocess(img_buffer)
        proc_h, proc_w = img_gray.shape[:2]
        img_hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)

        # 1. Feature Map Computations
        lbp_ent_map = self._compute_lbp_entropy_saliency(img_gray)
        gabor_map = self._compute_gabor_saliency(img_gray) if feature_mode in ["gabor", "all"] else None
        glcm_map = self._compute_glcm_saliency(img_gray, num_levels=GLCM_SETTINGS.get("NUM_LEVELS", 32)) if feature_mode in ["glcm", "all"] else None

        # 2. Channel Fusion
        if feature_mode == "lbp_entropy":
            saliency_map = lbp_ent_map
        elif feature_mode == "gabor":
            saliency_map = np.maximum(lbp_ent_map, gabor_map)
        elif feature_mode == "glcm":
            saliency_map = np.maximum(lbp_ent_map, glcm_map)
        else:  # "all"
            saliency_map = np.maximum(lbp_ent_map, np.maximum(gabor_map, glcm_map))

        # 3. Dynamic Auto-Calibrated Thresholding
        mean_sal = float(np.mean(saliency_map))
        std_sal = float(np.std(saliency_map))
        thresh_val = min(240, mean_sal + sensitivity * std_sal)
        _, binary = cv2.threshold(saliency_map, thresh_val, 255, cv2.THRESH_BINARY)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.defects = []
        min_area = TEXTURE_SETTINGS.get("MIN_DEFECT_AREA", 300)
        margin_px = GEOMETRY_SETTINGS.get("BORDER_MARGIN_PX", 8)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Suppress boundary artifacts
            if (x <= margin_px or y <= margin_px or (x + w) >= (proc_w - margin_px) or (y + h) >= (proc_h - margin_px)) and (area < min_area * 1.5):
                continue

            geom = extract_defect_geometry(contour, img_small, img_hsv, scale, mm_per_pixel=mm_per_pixel)

            aspect_ratio = geom["aspect_ratio"]
            circularity = geom["circularity"]
            delta_rgb = geom["delta_rgb"]

            # Defect taxonomy classification
            if aspect_ratio > 3.0 or aspect_ratio < 0.33:
                d_type = "Slub"
            elif circularity < 0.30:
                d_type = "Weave Irregularity"
            elif delta_rgb > 25.0:
                d_type = "Snag / Knot"
            else:
                d_type = "Weave Irregularity"

            # Confidence score based on z-distance in saliency map
            roi_sal = saliency_map[y:y + h, x:x + w]
            roi_mean = float(np.mean(roi_sal)) if roi_sal.size > 0 else mean_sal
            z_distance = abs(roi_mean - mean_sal) / max(std_sal, 1e-4)
            confidence = min(99, max(20, int((z_distance / max(sensitivity, 1e-4)) * 70)))

            self.defects.append({
                "Type": d_type,
                "Area (px)": geom["area_px"],
                "Solidity": f"{geom['circularity']:.2f}",
                "Quality Score": f"{confidence}%",
                "bbox_x": geom["bbox_x"],
                "bbox_y": geom["bbox_y"],
                "bbox_w": geom["bbox_w"],
                "bbox_h": geom["bbox_h"],
                "polygon": geom["polygon"],
                "Perimeter (px)": geom["perimeter_px"],
                "Circularity": geom["circularity"],
                "Aspect Ratio": geom["aspect_ratio"],
                "Major Axis (px)": geom["major_axis_px"],
                "Minor Axis (px)": geom["minor_axis_px"],
                "Delta RGB": geom["delta_rgb"],
                "Delta HSV": geom["delta_hsv"],
                "Area (mm2)": geom["area_mm2"],
                "Major Axis (mm)": geom["major_axis_mm"],
                "Major Axis (in)": geom["major_axis_in"],
                "Calibrated": geom["calibrated"],
                "Pipeline": "Texture",
                "Category": "Surface" if d_type in ["Slub", "Snag / Knot"] else "Structural",
            })

        viz_maps = {
            "texture_saliency_map": saliency_map,
            "grayscale_clahe": img_gray,
            "lbp_entropy_map": lbp_ent_map,
        }
        if gabor_map is not None:
            viz_maps["gabor_saliency_map"] = gabor_map
        if glcm_map is not None:
            viz_maps["glcm_anomaly_map"] = glcm_map

        return self.defects, viz_maps


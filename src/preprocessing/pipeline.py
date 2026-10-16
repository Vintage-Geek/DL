"""Channel-aware image preprocessing for controlled comparison experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

METHODS = (
    "P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "CLAHE"
)


def _as_float(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    if values.ndim not in (2, 3) or (values.ndim == 3 and values.shape[2] not in (1, 3)):
        raise ValueError("Expected a grayscale or RGB image with 1 or 3 channels.")
    if values.size == 0:
        raise ValueError("Image is empty.")
    if np.issubdtype(values.dtype, np.integer):
        return values.astype(np.float32) / float(np.iinfo(values.dtype).max)
    values = values.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Image contains NaN or infinite pixel values.")
    if values.min() < 0.0 or values.max() > 1.0:
        values = values / 255.0
    return np.clip(values, 0.0, 1.0)


def _restore_channels(image: np.ndarray, channels: int) -> np.ndarray:
    if channels == 3 and image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if channels == 3 and image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image[..., 0], cv2.COLOR_GRAY2RGB)
    if channels == 1 and image.ndim == 2:
        return image[..., None]
    return image


def _contrast_stretch(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, (1, 99))
    if high <= low:
        return image.copy()
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def _equalize_luminance(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2 or image.shape[2] == 1:
        plane = image if image.ndim == 2 else image[..., 0]
        result = cv2.equalizeHist(np.round(plane * 255).astype(np.uint8)).astype(np.float32) / 255
        return result if image.ndim == 2 else result[..., None]
    lab = cv2.cvtColor(np.round(image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.equalizeHist(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255


def preprocess_image(
    image: np.ndarray,
    method: str = "P0",
    *,
    grayscale_compatible: bool = False,
    clahe_clip_limit: float = 2.0,
    clahe_grid_size: int = 8,
) -> np.ndarray:
    """Apply one named method to RGB/grayscale pixels; output remains channel-last float32.

    Methods operate on pixels before ImageNet normalization. P3 is intentionally opt-in
    because grayscale conversion discards color information; RGB is restored for ResNet.
    P9 returns genuine per-image, per-channel z-scores and can be outside [0, 1].
    """
    if method not in METHODS:
        raise ValueError(f"Unknown preprocessing method {method!r}; choose from {METHODS}.")
    if clahe_clip_limit <= 0 or clahe_grid_size < 1:
        raise ValueError("CLAHE clip limit and grid size must be positive.")
    source = np.asarray(image)
    original_channels = 1 if source.ndim == 2 or source.shape[-1] == 1 else 3
    pixels = _as_float(source)
    if pixels.ndim == 2:
        pixels = pixels[..., None]

    if method == "P0":
        result = pixels
    elif method == "P1":
        result = np.clip(pixels * 1.15, 0.0, 1.0)
    elif method == "P2":
        result = _contrast_stretch(pixels)
    elif method == "P3":
        if not grayscale_compatible:
            raise ValueError("P3 grayscale processing is opt-in; set grayscale_compatible=true.")
        gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY) if pixels.shape[2] == 3 else pixels[..., 0]
        result = _contrast_stretch(gray)[..., None]
    elif method == "P4":
        result = _equalize_luminance(pixels[..., 0] if pixels.shape[2] == 1 else pixels)
        if result.ndim == 2:
            result = result[..., None]
    elif method == "P5":
        result = cv2.blur(pixels, (3, 3))
    elif method == "P6":
        result = cv2.medianBlur(pixels, 3)
    elif method == "P7":
        result = cv2.GaussianBlur(pixels, (0, 0), sigmaX=1.0, sigmaY=1.0)
    elif method == "P8":
        minimum = pixels.min(axis=(0, 1), keepdims=True)
        maximum = pixels.max(axis=(0, 1), keepdims=True)
        result = np.divide(pixels - minimum, maximum - minimum, out=np.zeros_like(pixels), where=maximum > minimum)
    elif method == "P9":
        mean = pixels.mean(axis=(0, 1), keepdims=True)
        standard_deviation = pixels.std(axis=(0, 1), keepdims=True)
        result = np.divide(pixels - mean, standard_deviation, out=np.zeros_like(pixels), where=standard_deviation > 1e-8)
    else:
        if pixels.shape[2] == 1:
            channel = pixels[..., 0]
            clahe = cv2.createCLAHE(clahe_clip_limit, (clahe_grid_size, clahe_grid_size))
            result = (clahe.apply(np.round(channel * 255).astype(np.uint8)).astype(np.float32) / 255)[..., None]
        else:
            lab = cv2.cvtColor(np.round(pixels * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clahe_clip_limit, (clahe_grid_size, clahe_grid_size))
            lab[..., 0] = clahe.apply(lab[..., 0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255

    result = np.asarray(result, dtype=np.float32)
    result = _restore_channels(result, original_channels)
    return np.ascontiguousarray(result)


@dataclass(frozen=True)
class PreprocessingPipeline:
    method: str = "P0"
    grayscale_compatible: bool = False
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return preprocess_image(
            image,
            self.method,
            grayscale_compatible=self.grayscale_compatible,
            clahe_clip_limit=self.clahe_clip_limit,
            clahe_grid_size=self.clahe_grid_size,
        )
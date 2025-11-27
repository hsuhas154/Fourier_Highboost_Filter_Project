# io/image_handler.py
"""
Image read/write helpers using Pillow.

Functions:
- read_image(path) -> numpy array (H x W) or (H x W x 3), dtype preserved
- save_image(path, array) -> writes image
- detect_is_color(array) -> bool
"""

from PIL import Image
import pillow_avif
import os
import numpy as np
from typing import Tuple


def read_image(path: str) -> Tuple[np.ndarray, dict]:
    """
    Read an image from `path` and return (array, meta).
    - Returns RGB arrays of shape (H,W,3) or grayscale (H,W).
    - Meta contains mode and size. If image has alpha, meta includes 'has_alpha' and meta['alpha'] as a separate array.
    """
    img = Image.open(path)
    mode = img.mode
    # Convert to a consistent representation: preserve alpha separately if present
    if mode in ("RGBA", "LA") or ("transparency" in img.info):
        img = img.convert("RGBA")
        arr = np.asarray(img)
        rgb = arr[..., :3]
        alpha = arr[..., 3]
        meta = {"mode": "RGBA", "size": img.size, "has_alpha": True}
        meta["alpha"] = alpha
        return rgb, meta
    else:
        # convert to RGB for color images, L for grayscale
        if mode.startswith("RGB") or mode in ("P",) or path.lower().endswith(".avif"):
            img = img.convert("RGB")
            arr = np.asarray(img)
            meta = {"mode": "RGB", "size": img.size, "has_alpha": False}
            return arr, meta
        else:
            img = img.convert("L")
            arr = np.asarray(img)
            meta = {"mode": "L", "size": img.size, "has_alpha": False}
            return arr, meta


def save_image(path: str, array: np.ndarray):
    """
    Save an image array to `path`. Accepts HxW (grayscale) or HxWx3 (RGB).
    Casts floats to uint8 by clipping to 0..255.
    """
    if array.ndim == 2:
        mode = "L"
    elif array.ndim == 3 and array.shape[2] == 3:
        mode = "RGB"
    else:
        raise ValueError("save_image expects HxW or HxWx3 array.")

    # Cast to uint8 if necessary
    if np.issubdtype(array.dtype, np.floating):
        arr = np.clip(array, 0.0, 255.0).astype(np.uint8)
    else:
        arr = array.astype(np.uint8)

    img = Image.fromarray(arr, mode=mode)
    img.save(path)


def detect_is_color(array: np.ndarray) -> bool:
    return array.ndim == 3 and array.shape[2] == 3


def _looks_like_satellite(path: str, arr: np.ndarray, size_threshold_bytes: int = 40 * 1024 * 1024) -> bool:
    """
    Heuristic to detect scientific / high-bit-depth rasters that need preview normalization.
    Returns True if arr/path likely represent a raster to normalize:
      - dtype is uint16/int16 OR
      - pixel values exceed 8-bit range (max > 255) OR
      - file size is large (> size_threshold_bytes)
    """
    try:
        # dtype-based detection (fast)
        if arr.dtype in (np.uint16, np.int16):
            return True
    except Exception:
        pass

    try:
        # value-range based detection
        if np.nanmax(arr) > 255:
            return True
    except Exception:
        # if arr is weird (e.g., ragged), skip this check
        pass

    try:
        # file size fallback (40 MB default)
        if os.path.exists(path) and os.path.getsize(path) > size_threshold_bytes:
            return True
    except Exception:
        pass

    return False


def normalize_to_uint8(arr: np.ndarray, clip_percentiles=(1, 99)) -> np.ndarray:
    """
    Normalize a numeric array to uint8 for display.
    Uses percentile stretch by default to reduce effect of outliers.
    Returns a uint8 array suitable for preview/display.
    """
    arrf = arr.astype(np.float32)
    arrf = np.nan_to_num(arrf, nan=0.0, posinf=np.nanmax(arrf), neginf=np.nanmin(arrf))
    p_low, p_high = clip_percentiles

    try:
        if p_low is not None and p_high is not None and 0 <= p_low < p_high <= 100:
            vmin = float(np.percentile(arrf, p_low))
            vmax = float(np.percentile(arrf, p_high))
        else:
            vmin = float(np.min(arrf))
            vmax = float(np.max(arrf))
    except Exception:
        vmin = float(np.min(arrf))
        vmax = float(np.max(arrf))

    if vmax <= vmin:
        # constant image: map to mid-gray or zeros
        out = np.clip(arrf - vmin, 0, 255)
        return out.astype(np.uint8)

    scaled = (arrf - vmin) / (vmax - vmin)
    scaled = (scaled * 255.0).clip(0, 255)
    return scaled.astype(np.uint8)

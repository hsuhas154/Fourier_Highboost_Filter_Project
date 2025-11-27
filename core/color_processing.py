"""
core/color_processing.py

Per-channel processing pipeline for grayscale and RGB images.
Implements the canonical frequency-domain high-boost pipeline:
  1) FFT
  2) fftshift (center DC)
  3) build low-pass mask L (via filters.build_lowpass_mask or caller-supplied mask)
  4) compose high-boost mask Hb
  5) multiply in shifted domain: Gs = Fs * Hb
  6) ifftshift(Gs)
  7) inverse FFT -> spatial output

API:
- process_grayscale(image, mask_params_or_fn, r, do_srgb_linearize=False, return_intermediates=False)
- process_color_rgb(image_rgb, mask_params_or_fn, r, do_srgb_linearize=False, return_intermediates=False)

mask_params_or_fn may be:
  - a callable mask_fn(shape, **kwargs)  (e.g., core.filters.gaussian_lowpass_mask)
  - or a dict with keys suitable for filters.build_lowpass_mask:
      {"mask_type": "gaussian", "D0": 30.0, "order": 2, "center": None, "align_with_fftshift": True}

If return_intermediates=True, the function returns (output_image, intermediates_dict)
where intermediates_dict contains keys: 'F', 'F_shifted', 'L', 'Hb', 'G_shifted', 'G'  (per-channel lists for color).
"""

from typing import Callable, Dict, Optional, Tuple, Any
import numpy as np

from .fft_engine import compute_fft, compute_ifft, fft_shift, ifft_shift, magnitude_spectrum
from . import filters
from .highboost import compose_highboost_mask, compose_highboost_from_mask_params

# --- sRGB helpers (same as previous implementations) ---
def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    a = 0.055
    img = np.clip(img, 0.0, 1.0)
    return np.where(img <= 0.04045, img / 12.92, ((img + a) / (1 + a)) ** 2.4)


def linear_to_srgb(img_lin: np.ndarray) -> np.ndarray:
    a = 0.055
    img_lin = np.clip(img_lin, 0.0, 1.0)
    return np.where(img_lin <= 0.0031308, img_lin * 12.92, (1 + a) * (img_lin ** (1.0 / 2.4)) - a)


# --- helpers for range/dtype preservation ---
def _to_float_and_range(arr: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Convert array to float64 and return original [min,max] range (based on dtype).
    For integer dtypes use full dtype range (e.g., 0..255). For floats infer min/max from data.
    """
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        return arr.astype(np.float64), float(info.min), float(info.max)
    else:
        a = arr.astype(np.float64)
        return a, float(np.nanmin(a)), float(np.nanmax(a))


def _clip_and_cast(arr: np.ndarray, dtype, vmin: float, vmax: float) -> np.ndarray:
    """
    Clip arr to [vmin, vmax] and cast to dtype.
    """
    out = np.clip(arr, vmin, vmax)
    if np.issubdtype(dtype, np.integer):
        return np.rint(out).astype(dtype)
    else:
        return out.astype(dtype)


# --- internal single-channel pipeline (explicit shifts) ---
def _process_channel(
    channel: np.ndarray,
    mask_builder: Any,
    mask_kwargs: Dict,
    r: float,
    *,
    do_srgb_linearize: bool = False,
    return_intermediates: bool = False,
):
    """
    Process one 2D channel array and apply high-boost filtering.

    mask_builder:
      - callable(shape, **mask_kwargs) -> L mask
      - or None: then mask_kwargs must be provided for compose_highboost_from_mask_params (expects D0 and mask_type)
    """
    if r is None or float(r) <= 1.0:
        raise ValueError("Boost factor r must be provided and > 1.")

    # preserve range & dtype info
    orig_dtype = channel.dtype
    ch_float, vmin, vmax = _to_float_and_range(channel)

    # optional sRGB linearization (assume integer inputs are 0..255)
    if do_srgb_linearize and np.issubdtype(orig_dtype, np.integer):
        # normalize to 0..1, linearize, then scale back to original numeric scale
        ch_norm = ch_float / float(vmax if vmax > 0 else 255.0)
        ch_lin = srgb_to_linear(ch_norm)
        ch_float = ch_lin * float(vmax)

    # 1: FFT
    F = compute_fft(ch_float)
    # 2: shift DC to center
    F_shifted = fft_shift(F)

    # prepare L (low-pass mask)
    shape = F_shifted.shape
    if callable(mask_builder):
        L = mask_builder(shape, **(mask_kwargs or {}))
    else:
        # assume mask_builder == None and mask_kwargs contains build_lowpass_mask params
        # allow passing e.g. {"mask_type":"gaussian","D0":30,"order":2,"center":None,"align_with_fftshift":True}
        L = filters.build_lowpass_mask(shape=shape, **(mask_kwargs or {}))

    # Defensive checks
    if L.shape != shape:
        raise ValueError("Low-pass mask shape does not match FFT shape.")

    # Compose high-boost mask Hb
    Hb = compose_highboost_mask(L, r)

    # Multiply in shifted domain
    G_shifted = F_shifted * Hb

    # inverse shift and inverse FFT
    G = ifft_shift(G_shifted)
    out_float = compute_ifft(G)

    # optional inverse sRGB gamma
    if do_srgb_linearize and np.issubdtype(orig_dtype, np.integer):
        # out_float currently in same numeric scale as ch_float (e.g., 0..255)
        out_norm = np.clip(out_float / float(vmax if vmax > 0 else 255.0), 0.0, 1.0)
        out_lin = linear_to_srgb(out_norm)
        out_float = out_lin * float(vmax)

    out = _clip_and_cast(out_float, orig_dtype, vmin, vmax)

    if return_intermediates:
        intermediates = {
            "F": F,
            "F_shifted": F_shifted,
            "L": L,
            "Hb": Hb,
            "G_shifted": G_shifted,
            "G": G,
            "out_float": out_float,
        }
        return out, intermediates

    return out


# --- public APIs ---
def process_grayscale(
    image: np.ndarray,
    mask_builder: Any,
    mask_kwargs: Dict,
    r: float,
    *,
    do_srgb_linearize: bool = False,
    return_intermediates: bool = False,
) -> Any:
    """
    Process a grayscale 2D image with a high-boost filter.

    mask_builder and mask_kwargs as described in _process_channel.
    """
    if image.ndim != 2:
        raise ValueError("process_grayscale expects a 2D array.")

    return _process_channel(
        image,
        mask_builder,
        mask_kwargs,
        r,
        do_srgb_linearize=do_srgb_linearize,
        return_intermediates=return_intermediates,
    )


def process_color_rgb(
    image_rgb: np.ndarray,
    mask_builder: Any,
    mask_kwargs: Dict,
    r: float,
    *,
    do_srgb_linearize: bool = False,
    return_intermediates: bool = False,
) -> Any:
    """
    Process an HxWx3 RGB image by applying the high-boost pipeline to each channel separately.

    Returns (output_image) or (output_image, intermediates) when return_intermediates=True.
    intermediates (if returned) is a dict with keys 'R','G','B' each mapping to that channel's intermediates dict.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("process_color_rgb expects an HxWx3 RGB image array.")

    chans_out = []
    inter_all = {"R": None, "G": None, "B": None}

    for idx in range(3):
        ch = image_rgb[:, :, idx]
        if return_intermediates:
            ch_out, ch_inter = _process_channel(
                ch,
                mask_builder,
                mask_kwargs,
                r,
                do_srgb_linearize=do_srgb_linearize,
                return_intermediates=True,
            )
            inter_all[["R", "G", "B"][idx]] = ch_inter
        else:
            ch_out = _process_channel(
                ch,
                mask_builder,
                mask_kwargs,
                r,
                do_srgb_linearize=do_srgb_linearize,
                return_intermediates=False,
            )
        chans_out.append(ch_out)

    stacked = np.stack(chans_out, axis=2)

    if return_intermediates:
        return stacked, inter_all
    return stacked
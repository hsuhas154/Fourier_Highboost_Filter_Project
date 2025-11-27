"""
visuals/plots.py

Plotting utilities for spectrum, masks, high-boost masks and comparisons.

APIs:
- plot_magnitude_spectrum(F, out_path=None, is_shifted=True, log=True, title=None)
- plot_mask(L, out_path=None, title=None, cmap=None)
- plot_highboost_mask(Hb, out_path=None, title=None, cmap=None)
- compare_and_save(original, boosted, spectrum=None, mask=None, Hb=None, out_path=None, titles=None)
- plot_channel_breakdown(channels, out_dir=None, base_name='channel', ext='png')
- fig_to_array(fig) -> np.ndarray (H,W,3) uint8

Notes:
- This module uses matplotlib. It does not modify core behavior.
- If out_path is None, functions will return the matplotlib Figure object (caller can save or display).
"""

from typing import Optional, Sequence, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from core.fft_engine import magnitude_spectrum, fft_shift
from io_utils.file_utils import make_result_filename

# Helper to ensure outdir exists
def _ensure_outdir(out_path: Optional[str]):
    if out_path is None:
        return None
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    return out_path

def _save_raw_array_image(out_path: Optional[str], arr: np.ndarray, log_scale: bool = False):
    """
    Save a numeric array as a raw image (PNG). No Matplotlib involved.
    - arr can be 2D (scalar) or 3D (H,W,3). For scalar arrays, map to uint8 [0,255].
    - log_scale: apply log1p before normalization (useful for magnitude spectrum).
    """
    if out_path is None:
        return None

    _ensure_outdir(out_path)

    a = np.array(arr, copy=True)  # ensure numpy array

    # If complex, use magnitude
    if np.iscomplexobj(a):
        a = np.abs(a)

    # Optionally apply log scaling
    if log_scale:
        # add tiny eps to avoid log(0)
        a = np.log1p(np.abs(a))

    # Normalize to 0..255
    amin = float(np.nanmin(a))
    amax = float(np.nanmax(a))
    if np.isfinite(amin) and np.isfinite(amax) and amax > amin:
        norm = (a - amin) / (amax - amin)
    else:
        # fallback: zeros
        norm = np.zeros_like(a, dtype=np.float32)

    # Convert to uint8
    if norm.ndim == 2:
        img_arr = (np.clip(norm * 255.0, 0, 255)).astype(np.uint8)
        img = Image.fromarray(img_arr, mode='L')
    elif norm.ndim == 3 and norm.shape[2] == 3:
        img_arr = (np.clip(norm * 255.0, 0, 255)).astype(np.uint8)
        img = Image.fromarray(img_arr, mode='RGB')
    else:
        # squeeze any extra dims
        arr2 = np.squeeze(norm)
        if arr2.ndim == 2:
            img_arr = (np.clip(arr2 * 255.0, 0, 255)).astype(np.uint8)
            img = Image.fromarray(img_arr, mode='L')
        else:
            # as a last resort convert to uint8 and save raw bytes
            img = Image.fromarray((np.clip(norm,0,1)*255).astype(np.uint8))

    img.save(out_path)
    return out_path

def fig_to_array(fig: plt.Figure, dpi: int = 100) -> np.ndarray:
    """
    Convert a Matplotlib figure to an HxWx3 uint8 RGB numpy array.
    """
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    arr = buf.reshape((h, w, 3))
    return arr

def _save_or_return(fig: plt.Figure, out_path: Optional[str]):
    if out_path is not None:
        _ensure_outdir(out_path)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path
    else:
        return fig

def _prepare_spectrum_for_display(F: np.ndarray, is_shifted: bool = True, log: bool = True) -> np.ndarray:
    """
    Return a display-friendly 2D float array for the magnitude spectrum.
    """
    if not is_shifted:
        # center it for display (don't change original)
        F_disp = fft_shift(F)
    else:
        F_disp = F
    mag = magnitude_spectrum(F_disp, log=log)
    # Normalize to [0,1] for colormap consistency
    mag = mag - float(np.nanmin(mag))
    denom = float(np.nanmax(mag)) if float(np.nanmax(mag)) != 0.0 else 1.0
    mag = mag / denom
    return mag

def plot_magnitude_spectrum(
    F: np.ndarray,
    out_path: Optional[str] = None,
    is_shifted: bool = True,
    log: bool = True,
    title: Optional[str] = "Magnitude Spectrum",
) -> Optional[str]:
    """
    Save raw magnitude spectrum image (no Matplotlib when out_path given).
    If out_path provided -> write raw PNG using _save_raw_array_image (log-scaled by default).
    If out_path is None -> fall back to returning the prepared 2D array (not a figure).
    """
    if not is_shifted:
        F_disp = fft_shift(F)
    else:
        F_disp = F

    mag = np.abs(F_disp)

    # If out_path specified, write raw image (optionally log-scale controlled by `log`)
    if out_path is not None:
        return _save_raw_array_image(out_path, mag, log_scale=bool(log))

    # otherwise return a normalized 2D float array for caller to display if desired
    magf = mag - float(np.nanmin(mag))
    denom = float(np.nanmax(magf)) if float(np.nanmax(magf)) != 0.0 else 1.0
    magf = magf / denom
    return magf

def plot_phase_spectrum(
    F: np.ndarray,
    out_path: Optional[str] = None,
    is_shifted: bool = True,
    title: Optional[str] = "Phase Spectrum",
) -> Optional[str]:
    """
    Save raw phase spectrum image (no Matplotlib when out_path given).
    Phase is mapped from -pi..pi -> 0..1 before saving.
    If out_path provided -> write raw PNG using _save_raw_array_image.
    If out_path is None -> return normalized 2D float array.
    """
    if not is_shifted:
        F_disp = fft_shift(F)
    else:
        F_disp = F

    # compute phase in radians (-pi..pi)
    phase = np.angle(F_disp)

    # normalize to 0..1 (linear)
    phase_norm = (phase + np.pi) / (2.0 * np.pi)

    if out_path is not None:
        # save as raw grayscale image (0..1 -> 0..255)
        return _save_raw_array_image(out_path, phase_norm, log_scale=False)

    # fallback: return normalized array for interactive use
    return phase_norm

def plot_mask(
    L: np.ndarray,
    out_path: Optional[str] = None,
    title: Optional[str] = "Low-pass Mask",
    cmap: Optional[str] = None,
) -> Optional[str]:
    """
    Save raw low-pass mask as grayscale PNG (no Matplotlib when out_path given).
    L expected in 0..1 but function will normalize whatever range is present.
    """
    if out_path is not None:
        return _save_raw_array_image(out_path, L, log_scale=False)

    # fallback: return normalized array for interactive use
    a = np.array(L, dtype=np.float64)
    a = a - float(np.nanmin(a))
    denom = float(np.nanmax(a)) if float(np.nanmax(a)) != 0.0 else 1.0
    a = a / denom
    return a

def plot_highboost_mask(
    Hb: np.ndarray,
    out_path: Optional[str] = None,
    title: Optional[str] = "High-Boost Mask",
    cmap: Optional[str] = None,
) -> Optional[str]:
    """
    Save raw high-boost mask as grayscale PNG (no Matplotlib when out_path given).
    Hb may exceed 1.0; we normalize to its min/max when writing.
    """
    if out_path is not None:
        return _save_raw_array_image(out_path, Hb, log_scale=False)

    # fallback: return normalized array for interactive use
    a = np.array(Hb, dtype=np.float64)
    a = a - float(np.nanmin(a))
    denom = float(np.nanmax(a)) if float(np.nanmax(a)) != 0.0 else 1.0
    a = a / denom
    return a

def compare_and_save(
    original: np.ndarray,
    boosted: np.ndarray,
    spectrum: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    Hb: Optional[np.ndarray] = None,
    out_path: Optional[str] = None,
    titles: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """
    SAVE ONLY:
    Original (left) | High-Boosted (right)
    High-resolution figure.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # HD WIDE output

    # --- ORIGINAL ---
    ax = axs[0]
    if original.ndim == 2:
        ax.imshow(original, cmap="gray", interpolation="nearest")
    else:
        ax.imshow(original.astype(np.uint8))
    ax.set_title("Original")
    ax.axis("off")

    # --- BOOSTED ---
    ax = axs[1]
    if boosted.ndim == 2:
        ax.imshow(boosted, cmap="gray", interpolation="nearest")
    else:
        ax.imshow(boosted.astype(np.uint8))
    ax.set_title("High-Boosted")
    ax.axis("off")

    # SAVE
    if out_path:
        _ensure_outdir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        return out_path
    else:
        return fig


def plot_channel_breakdown(
    channels: Sequence[np.ndarray],
    out_dir: Optional[str] = None,
    base_name: str = "channel",
    ext: str = "png",
) -> Tuple[Optional[str], list]:
    """
    Save each channel (list or tuple of 3 arrays) to out_dir with deterministic names.
    Returns (last_saved_path_or_None, list_of_paths).
    """
    paths = []
    if out_dir is None:
        # just return figures
        for i, ch in enumerate(channels):
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(ch, origin="upper")
            ax.axis("off")
            paths.append(fig)
        return None, paths

    os.makedirs(out_dir, exist_ok=True)
    for i, ch in enumerate(channels):
        name = f"{base_name}_{i}.{ext}"
        p = os.path.join(out_dir, name)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(ch, origin="upper")
        ax.axis("off")
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths[-1] if paths else None, paths

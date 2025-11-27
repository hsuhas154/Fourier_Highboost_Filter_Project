"""
core/highboost.py

High-boost composition and application utilities

Provided functions:
- compose_highboost_mask(L, r)
- apply_highboost_to_frequency(F, Hb)
- compose_highboost_from_mask_params(shape, D0, mask_type='gaussian', order=2, center=None, align_with_fftshift=True, r=1.5)

Notes:
- r is required to be > 1 (project specification). Functions validate this and raise ValueError otherwise.
- compose_highboost_from_mask_params() is a convenience wrapper that builds a low-pass mask
  using filters.build_lowpass_mask(...) and returns the high-boost mask Hb aligned to the given shape.
  The returned Hb may be used directly with an FFT array F (elementwise multiplication).
"""

from typing import Tuple, Optional
import numpy as np
from core import filters

def compose_highboost_mask(L: np.ndarray, r: float) -> np.ndarray:
    """
    Compose a high-boost mask Hb from a low-pass mask L and boost factor r.
    Hb(u,v) = r + (1 - r) * L(u,v)

    Parameters
    ----------
    L : np.ndarray
        Low-pass mask (numeric array). Expected values in [0, 1], but values will be clipped defensively.
    r : float
        Boost factor. Must be > 1 per project specification.

    Returns
    -------
    np.ndarray
        High-boost mask Hb, same shape as L and dtype=float64.
    """
    if r is None:
        raise ValueError("Boost factor r must be provided.")
    r = float(r)
    if r <= 1.0:
        raise ValueError("Boost factor r must be > 1 according to the project specification.")
    # Defensive: ensure numeric array, cast to float
    L_arr = np.asarray(L, dtype=float)
    # Clip L to [0,1] to be robust against invalid inputs
    Lc = np.clip(L_arr, 0.0, 1.0)
    Hb = r + (1.0 - r) * Lc
    # --- IMPORTANT: enforce exact geometrical symmetry ---
    # Average with its 180-degree rotation (flip both axes) so Hb[i,j] == Hb[-i,-j] exactly.
    Hb = 0.5 * (Hb + Hb[::-1, ::-1])
    return Hb.astype(np.float64)

def apply_highboost_to_frequency(F: np.ndarray, Hb: np.ndarray) -> np.ndarray:
    """
    Apply a real-valued high-boost mask Hb elementwise to a complex frequency array F.
    Performs shape validation and returns the complex-valued result.

    Parameters
    ----------
    F : np.ndarray
        Complex frequency-domain array (2D).
    Hb : np.ndarray
        Real-valued high-boost mask array (same shape as F).

    Returns
    -------
    np.ndarray
        Complex-valued frequency-domain result G = F * Hb.
    """
    if F.shape != Hb.shape:
        raise ValueError("F and Hb must have the same shape for elementwise multiplication.")
    # Allow Hb to be real dtype; cast to same dtype-broadcast compatible type
    return F * Hb

def compose_highboost_from_mask_params(
    shape: Tuple[int, int],
    D0: float,
    mask_type: str = "gaussian",
    order: int = 2,
    center: Optional[Tuple[float, float]] = None,
    align_with_fftshift: bool = True,
    r: float = 1.5,
) -> np.ndarray:
    """
    Convenience builder that creates a low-pass mask using filters.build_lowpass_mask(...)
    and returns the corresponding high-boost mask Hb for the given shape and parameters.
    This helper enforces the same validation rules (D0 >= 0, integer order >= 1, r > 1)
    by delegating to the filters.build_lowpass_mask and to compose_highboost_mask.

    Parameters
    ----------
    shape : tuple[int,int]
        Shape of the frequency array (M, N). The produced mask will match this shape.
    D0 : float
        Cutoff parameter (may be 0 for the explicit pinhole policy).
    mask_type : str
        One of {"radial", "gaussian", "butterworth"}.
    order : int
        Butterworth order (used only when mask_type == "butterworth").
    center : Optional[tuple[float,float]]
        Float-valued center coordinates if the user wishes to override canonical center.
    align_with_fftshift : bool
        If True (default), choose the pinhole index for D0==0 so it aligns with np.fft.fftshift DC location
        for even-sized axes (policy implemented in filters._choose_pinhole_index()).
    r : float
        Boost factor (> 1).

    Returns
    -------
    np.ndarray
        High-boost mask Hb (float64) shaped (M, N).
    """
    # Validation for D0 and order happens inside filters.build_lowpass_mask
    L = filters.build_lowpass_mask(
        shape=shape,
        D0=D0,
        mask_type=mask_type,
        order=order,
        center=center,
        align_with_fftshift=align_with_fftshift,
    )
    Hb = compose_highboost_mask(L, r)
    return Hb
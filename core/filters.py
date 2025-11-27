import numpy as np
from typing import Tuple, Optional

# --- Distance grid & helpers ---
def _distance_grid(
        shape: Tuple[int, int], 
        center: Optional[Tuple[float, float]] = None, 
        dtype=np.float64
    ) -> np.ndarray:
    """
    Build Euclidean distance grid D[u,v] from a float center.
    Default center = ((M-1)/2.0, (N-1)/2.0).
    """
    M, N = shape
    if center is None:
        u0 = (M - 1) / 2.0
        v0 = (N - 1) / 2.0
    else:
        u0, v0 = float(center[0]), float(center[1])
    u = np.arange(M, dtype=dtype).reshape(M, 1)
    v = np.arange(N, dtype=dtype).reshape(1, N)
    # Use hypot for numerical clarity and broadcasting issues
    # Could have used D = np.sqrt((u - u0) ** 2 + (v - v0) ** 2) too, but np.hypot() is better
    D = np.hypot(u - u0, v - v0)
    return D

def _choose_pinhole_index(
        shape: Tuple[int, int], 
        center: Optional[Tuple[float, float]] = None
    ) -> Tuple[int, int]:
    """
    Deciding which integer pixel to use for a 'pinhole' when D0==0.
    Methodology:
      - If center is provided (float coordinates), choose the nearest pixel (argmin on D).
      - If center is None:
          - For even-sized axes, choose (M//2, N//2) so it aligns with fftshift DC.
          - For odd-sized axes, choose ((M-1)//2, (N-1)//2) (the true middle pixel).
    Returns (row_index, col_index).
    """
    M, N = shape
    if center is not None:
        D = _distance_grid(shape, center=center)
        idx = np.unravel_index(int(np.argmin(D)), D.shape)
        return (int(idx[0]), int(idx[1]))
    # center was None -> apply even-sized or odd-sized methods
    row = (M // 2) if (M % 2 == 0) else ((M - 1) // 2)
    col = (N // 2) if (N % 2 == 0) else ((N - 1) // 2)
    return (int(row), int(col))

# --- Masks: radial, gaussian, butterworth ---
def radial_lowpass_mask(
        shape: Tuple[int, int], 
        D0: float, 
        center: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
    """
    Ideal radial (circular) low-pass mask.
    Explicitly handles D0 == 0 with _choose_pinhole_index (no NaN hacks).
    """
    if D0 < 0:
        raise ValueError("D0 must be non-negative for radial mask.")
    if float(D0) == 0.0:
        mask = np.zeros(shape, dtype=float)
        idx = _choose_pinhole_index(shape, center)
        mask[idx] = 1.0
        return mask
    D = _distance_grid(shape, center=center)
    mask = (D <= float(D0)).astype(float)
    return mask.astype(float)

def gaussian_lowpass_mask(
        shape: Tuple[int, int], 
        D0: float, 
        center: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
    """
    Gaussian low-pass: exp(-D^2 / (2*D0^2)).
    Explicit pinhole for D0 == 0. Numerically robust for tiny/large D0.
    """
    if D0 < 0:
        raise ValueError("D0 must be non-negative for Gaussian mask.")
    if float(D0) == 0.0:
        mask = np.zeros(shape, dtype=float)
        idx = _choose_pinhole_index(shape, center)
        mask[idx] = 1.0
        return mask
    D = _distance_grid(shape, center=center)
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = 2.0 * (float(D0) ** 2)
        arg = -(D**2) / denominator
        mask = np.exp(arg)
    mask = np.nan_to_num(mask, nan=1.0, posinf=1.0, neginf=0.0)
    mask = np.clip(mask, 0.0, 1.0)
    return mask.astype(float)

def butterworth_lowpass_mask(
        shape: Tuple[int, int], 
        D0: float, order: int = 2, 
        center: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
    """
    Butterworth low-pass mask: 1 / (1 + (D/D0)^(2*order)).
    Explicit pinhole for D0 == 0. Handles numeric overflow for large exponents.
    """
    if D0 < 0:
        raise ValueError("D0 must be non-negative for Butterworth mask.")
    if order < 1 or int(order) != order:
        raise ValueError("Butterworth order must be an integer, greater than or equal to 1.")
    order = int(order)
    if float(D0) == 0.0:
        mask = np.zeros(shape, dtype=float)
        idx = _choose_pinhole_index(shape, center)
        mask[idx] = 1.0
        return mask
    D = _distance_grid(shape, center=center)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = (D / float(D0)) ** (2 * order)
        mask = 1.0 / (1.0 + ratio)
    mask = np.nan_to_num(mask, nan=1.0, posinf=0.0, neginf=0.0)
    mask = np.clip(mask, 0.0, 1.0)
    return mask.astype(float)

# --- Convenience builder that applies the recommended pinhole methodology ---
def build_lowpass_mask(
    shape: Tuple[int, int],
    D0: float,
    mask_type: str = "gaussian",   # "radial", "gaussian", "butterworth"
    order: int = 2,
    center: Optional[Tuple[float, float]] = None,
    align_with_fftshift: bool = True
) -> np.ndarray:
    """
    Build a low-pass mask convenient for users.
    If center is None:
      - float center for distance grid = ((M-1)/2, (N-1)/2)
      - pinhole index chosen so that for even axes, when align_with_fftshift=True,
        pinhole sits at (M//2, N//2) to match np.fft.fftshift DC location.
    D0 == 0 is explicitly handled (single pixel at chosen index).
    """
    # Basic validation
    if D0 < 0:
        raise ValueError("D0 must be non-negative.")

    # floating center used to compute distances
    float_center = ((shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0) if center is None else (float(center[0]), float(center[1]))

    # decide pinhole index according to policy
    if center is not None:
        pinhole_idx = _choose_pinhole_index(shape, center=float_center)
    else:
        if align_with_fftshift:
            pinhole_idx = _choose_pinhole_index(shape, center=None)
        else:
            pinhole_idx = _choose_pinhole_index(shape, center=float_center)

    # D0 == 0 explicit pinhole
    if float(D0) == 0.0:
        mask = np.zeros(shape, dtype=float)
        mask[pinhole_idx] = 1.0
        return mask

    # Build requested mask using float_center (canonical)
    mask_type = mask_type.lower()
    if mask_type == "radial":
        return radial_lowpass_mask(shape, D0=D0, center=float_center)
    elif mask_type == "gaussian":
        return gaussian_lowpass_mask(shape, D0=D0, center=float_center)
    elif mask_type == "butterworth":
        return butterworth_lowpass_mask(shape, D0=D0, order=order, center=float_center)
    else:
        raise ValueError(f"Unknown mask_type '{mask_type}'. Choose 'radial', 'gaussian' or 'butterworth'.")

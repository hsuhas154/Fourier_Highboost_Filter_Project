'''
FFT engine helpers.

Functions:
- compute_fft: compute 2D FFT of a single-channel image (returns complex array)
- compute_ifft: inverse FFT, returns real part
- fft_shift / ifft_shift: wrappers around np.fft.fftshift / ifftshift
- magnitude_spectrum: log-scaled magnitude for visualization
'''

import numpy as np
import warnings
from typing import Tuple, Optional

def compute_fft(image: np.ndarray) -> np.ndarray:
    """
    Compute 2D FFT of an image (single channel, 2D object).
    Raises ValueError for non-2D inputs.
    """
    if image.ndim != 2:
        raise ValueError("compute_fft expects a 2D grayscale array.")
    return np.fft.fft2(image)

def compute_ifft(F: np.ndarray, imag_tol: float = 1e-9, suppress_warning: bool = True) -> np.ndarray:
    """
    Compute inverse 2D FFT and return real part.
    Warns if imaginary part is larger than imag_tol.
    """
    if F.ndim != 2:
        raise ValueError("compute_ifft expects a 2D frequency-domain array.")
    img_back = np.fft.ifft2(F)
    imag_max = float(np.max(np.abs(np.imag(img_back))))
    if not suppress_warning and imag_max > imag_tol:
        warnings.warn(
            f"Inverse FFT has non-negligible imaginary component (max abs = {imag_max}). "
            "Returning real part but consider checking your frequency-domain input.",
            RuntimeWarning
        )
    return np.real(img_back)

def fft_shift(F: np.ndarray) -> np.ndarray:
    """Shift zero-frequency to center (wrapper)."""
    return np.fft.fftshift(F)

def ifft_shift(Fs: np.ndarray) -> np.ndarray:
    """Inverse shift (center -> origin) (wrapper)."""
    return np.fft.ifftshift(Fs)

def magnitude_spectrum(F: np.ndarray, log: bool = True, eps: float = 1e-8) -> np.ndarray:
    """
    Return magnitude spectrum for visualization.
    If log is True, returns log1p(abs(F)+eps).
    """
    mag = np.abs(F)
    if log:
        return np.log1p(mag + eps)
    return mag

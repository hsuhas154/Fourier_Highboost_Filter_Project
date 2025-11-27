import numpy as np
import pytest
from core.fft_engine import (
    compute_fft, compute_ifft, fft_shift, ifft_shift, magnitude_spectrum
)

def test_fft_input_validation():
    arr = np.zeros((16, 16, 3))
    with pytest.raises(ValueError):
        compute_fft(arr)

def test_fft_ifft_roundtrip():
    img = np.random.rand(64, 64)
    F = compute_fft(img)
    back = compute_ifft(F)
    assert back.shape == img.shape
    assert np.allclose(img, back, atol=1e-8)

def test_ifft_imag_is_small():
    img = np.random.rand(32, 32)
    F = compute_fft(img)
    back = compute_ifft(F, suppress_warning=False)
    imag_max = np.max(np.abs(np.imag(np.fft.ifft2(F))))
    assert imag_max < 1e-7

def test_magnitude_spectrum_basic():
    img = np.random.rand(32, 32)
    F = compute_fft(img)
    mag = magnitude_spectrum(F, log=True)
    assert mag.shape == img.shape
    assert np.all(mag >= 0)

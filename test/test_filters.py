import numpy as np
import pytest
from core.filters import (
    radial_lowpass_mask, gaussian_lowpass_mask, butterworth_lowpass_mask,
    build_lowpass_mask, _distance_grid
)

def test_radial_mask_basic():
    L = radial_lowpass_mask((32, 32), D0=10)
    assert L.shape == (32, 32)
    assert np.all((L == 0) | (L == 1))

def test_gaussian_mask_values():
    L = gaussian_lowpass_mask((32, 32), D0=15)
    assert L.shape == (32, 32)
    assert np.all(L >= 0) and np.all(L <= 1)
    assert L[16,16] > L[0,0]  # center strongest

def test_butterworth_order_effect():
    L1 = butterworth_lowpass_mask((64, 64), D0=10, order=1)
    L4 = butterworth_lowpass_mask((64, 64), D0=10, order=4)
    # 1) central area should be near 1.0 for both (sanity)
    M, N = L1.shape
    cr, cc = M//2, N//2
    assert L1[cr, cc] > 0.9
    assert L4[cr, cc] > 0.9
    # 2) outside the cutoff (radius > D0) the higher-order mask must be smaller (sharper rolloff).
    D = _distance_grid(L1.shape)
    outside_idx = D > (10 + 3.0)  # a small margin beyond D0
    mean_out_L1 = float(L1[outside_idx].mean())
    mean_out_L4 = float(L4[outside_idx].mean())
    assert mean_out_L4 <= mean_out_L1 + 1e-12


def test_build_mask_D0_zero_pinhole_even():
    shape = (16, 16)
    L = build_lowpass_mask(shape=shape, D0=0, mask_type="gaussian")
    assert L.sum() == 1
    assert L[8, 8] == 1

def test_build_mask_center_override():
    L = build_lowpass_mask((16, 16), D0=0, center=(7.2, 6.8))
    assert L.sum() == 1  # pinhole selection still valid

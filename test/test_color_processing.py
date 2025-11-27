# test/test_color_processing.py
import numpy as np
import pytest
from core import filters
from core.color_processing import process_grayscale, process_color_rgb, srgb_to_linear, linear_to_srgb

def test_r_validation_grayscale():
    img = np.zeros((16, 16), dtype=np.uint8)
    with pytest.raises(ValueError):
        process_grayscale(img, mask_builder=None, mask_kwargs={"mask_type":"gaussian","D0":5}, r=1.0)

def test_pinhole_D0_zero_policy_even():
    shape = (16, 16)
    mask = filters.build_lowpass_mask(shape=shape, D0=0.0, mask_type="gaussian", align_with_fftshift=True)
    assert mask.sum() == 1.0
    # chosen index should match _choose_pinhole_index method (center for even dims)
    # For even dims method choices (M//2, N//2)
    assert mask[shape[0]//2, shape[1]//2] == 1.0

def test_pinhole_D0_zero_policy_odd():
    shape = (15, 13)
    mask = filters.build_lowpass_mask(shape=shape, D0=0.0, mask_type="gaussian", align_with_fftshift=True)
    assert mask.sum() == 1.0
    midr = (shape[0]-1)//2
    midc = (shape[1]-1)//2
    assert mask[midr, midc] == 1.0

def test_color_processing_runs_and_shapes():
    H, W = 32, 32
    # checkerboard
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[::2, ::2] = 255
    img[1::2, 1::2] = 255
    out = process_color_rgb(img, mask_builder=None, mask_kwargs={"mask_type":"gaussian","D0":5}, r=1.5)
    assert out.shape == img.shape
    assert out.dtype == img.dtype

def test_srgb_roundtrip():
    x = np.array([0.0, 0.02, 0.2, 0.5, 1.0])
    y = srgb_to_linear(x)
    z = linear_to_srgb(y)
    assert np.allclose(x, z, atol=1e-7)
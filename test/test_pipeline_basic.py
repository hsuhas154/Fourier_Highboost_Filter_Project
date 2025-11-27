import numpy as np
from core.color_processing import process_grayscale, process_color_rgb
from core.filters import build_lowpass_mask

def test_pipeline_constant_image():
    img = np.full((64,64), 150, dtype=np.uint8)
    out = process_grayscale(img, None, {"mask_type":"gaussian","D0":10}, r=1.5)
    assert np.allclose(out, img)  # constant images unchanged

def test_pipeline_edge_enhancement_float():
    # Use float image in range 0..1 to avoid integer clipping/saturation
    H, W = 64, 64
    img = np.zeros((H, W), dtype=np.float64)
    img[:, W//2 :] = 1.0  # left 0.0, right 1.0 (single vertical edge)

    # Use r=2.0 (still fine for float inputs, no clipping)
    out = process_grayscale(img, None, {"mask_type":"gaussian","D0":8}, r=2.0)

    def grad_mag_sum(a):
        a = a.astype(np.float64)
        gx = np.abs(np.diff(a, axis=1)).sum()
        gy = np.abs(np.diff(a, axis=0)).sum()
        return gx + gy

    original_grad = grad_mag_sum(img)
    boosted_grad = grad_mag_sum(out)

    # For floating test input, boosted gradient should be greater or equal
    assert boosted_grad >= original_grad - 1e-9
    # Ensure some numerical change occurred
    assert not np.allclose(out, img, atol=1e-9)


def test_color_pipeline_channel_independence():
    H, W = 64, 64
    img = np.zeros((H,W,3), dtype=np.uint8)
    img[...,0] = 200  # only red channel active
    out = process_color_rgb(img, None, {"mask_type":"gaussian","D0":10}, r=1.5)
    assert out[...,1].max() == 0
    assert out[...,2].max() == 0

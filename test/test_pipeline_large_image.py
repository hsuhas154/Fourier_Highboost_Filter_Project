# test/test_pipeline_large_image.py
import numpy as np
from core.color_processing import process_grayscale

def test_large_pipeline():
    img = (np.random.rand(512,512)*255).astype(np.uint8)
    out = process_grayscale(img, None, {"mask_type":"gaussian","D0":20}, r=1.5)
    assert out.shape == img.shape
    assert out.dtype == np.uint8

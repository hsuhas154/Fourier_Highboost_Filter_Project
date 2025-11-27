import numpy as np
import os
from io_utils.image_handler import read_image, save_image, detect_is_color

def test_detect_is_color():
    assert detect_is_color(np.zeros((16,16,3)))
    assert not detect_is_color(np.zeros((16,16)))

def test_save_and_read_roundtrip(tmp_path):
    arr = np.arange(100).reshape(10, 10).astype(np.uint8)
    p = tmp_path / "test.png"
    save_image(str(p), arr)
    out, meta = read_image(str(p))
    assert out.shape == arr.shape
    assert out.dtype == np.uint8

def test_read_image_alpha(tmp_path):
    from PIL import Image
    arr = np.zeros((10,10,4), dtype=np.uint8)
    p = tmp_path / "rgba.png"
    Image.fromarray(arr, mode="RGBA").save(p)
    rgb, meta = read_image(str(p))
    assert meta["has_alpha"]
    assert rgb.shape == (10,10,3)

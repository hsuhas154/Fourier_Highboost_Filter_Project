import os
import numpy as np
from io_utils.image_handler import read_image
from core.color_processing import process_grayscale

# --- Config ---
IMG_PATH = "data/sample1.png"   # or whichever image you want
D0 = 10.0
r = 1.8

# --- Read and convert to grayscale ---
arr, meta = read_image(IMG_PATH)
if arr.ndim == 3:
    img = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
else:
    img = arr

# --- Run the grayscale pipeline with intermediates ---
out, inter = process_grayscale(
    img,
    mask_builder=None,
    mask_kwargs={"mask_type": "gaussian", "D0": D0},
    r=r,
    return_intermediates=True,
)

# --- Diagnostics: symmetry & imaginary components ---
print("\n=== DEBUG INTERMEDIATES ===")
print(f"max imag in G (after ifft_shift): {np.max(np.abs(np.imag(inter['G']))):.6f}")
if np.iscomplexobj(inter['out_float']):
    print(f"max imag in out_float (before cast): {np.max(np.abs(np.imag(inter['out_float']))):.6f}")
else:
    print("out_float is real type")

Hb = inter['Hb']
print(f"Hb min/max: {Hb.min():.3f}, {Hb.max():.3f}")
print("Hb symmetric (flip both axes)?", np.allclose(Hb, Hb[::-1, ::-1]))

F_shifted = inter['F_shifted']
print("F_shifted Hermitian symmetry?",
      np.allclose(F_shifted, np.conj(np.flipud(np.fliplr(F_shifted)))))

G_shifted = inter['G_shifted']
print("G_shifted Hermitian symmetry?",
      np.allclose(G_shifted, np.conj(np.flipud(np.fliplr(G_shifted)))))

print("============================\n")

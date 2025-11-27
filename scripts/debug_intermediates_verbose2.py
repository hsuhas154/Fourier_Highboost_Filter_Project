# scripts/debug_intermediates_verbose2.py
import os
import numpy as np
from io_utils.image_handler import read_image
from core.color_processing import process_grayscale
from core.fft_engine import compute_fft, fft_shift

IMG_PATH = "data/sample1.png"   # change to your image path if needed
D0 = 10.0
r = 1.8

arr, meta = read_image(IMG_PATH)
if arr.ndim == 3:
    img = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
else:
    img = arr

print("img dtype, shape, min/max:", img.dtype, img.shape, img.min(), img.max())
print("any NaN/Inf in image?", np.isnan(img).any(), np.isinf(img).any())

F = compute_fft(img.astype(np.float64))
Fs = fft_shift(F)

print("F dtype:", F.dtype)
print("F real min/max:", np.min(F.real), np.max(F.real))
print("F imag min/max:", np.min(F.imag), np.max(F.imag))
print("any NaN/Inf in F?", np.isnan(F).any(), np.isinf(F).any())

# Correct Hermitian check using modulo mapping:
def hermitian_check_correct(A):
    M, N = A.shape
    # compute index arrays for -k mod N
    i_idx = (-np.arange(M)) % M
    j_idx = (-np.arange(N)) % N
    # build the conjugate-flipped array using fancy indexing
    B = np.conj(A[i_idx[:, None], j_idx[None, :]])
    diff = A - B
    max_abs = np.max(np.abs(diff))
    idx = np.unravel_index(np.argmax(np.abs(diff)), A.shape)
    return max_abs, idx, A[idx], B[idx], diff[idx]

max_unshifted, idx_u, Au, Bu, du = hermitian_check_correct(F)
max_shifted, idx_s, As, Bs, ds = hermitian_check_correct(Fs)
print(f"Hermitian max diff (unshifted): {max_unshifted}, idx={idx_u}, A[idx]={Au}, conj_target[idx]={Bu}")
print(f"Hermitian max diff (shifted):   {max_shifted}, idx={idx_s}, A[idx]={As}, conj_target[idx]={Bs}")

# show neighborhood around worst shifted idx
r_neigh = 2
i0, j0 = idx_s
print("Neighborhood (real part) around worst shifted idx:")
print(np.round(Fs.real[i0-r_neigh:i0+r_neigh+1, j0-r_neigh:j0+r_neigh+1], 6))
print("Neighborhood (imag part) around worst shifted idx:")
print(np.round(Fs.imag[i0-r_neigh:i0+r_neigh+1, j0-r_neigh:j0+r_neigh+1], 6))

# pipeline intermediates
out, inter = process_grayscale(
    img,
    mask_builder=None,
    mask_kwargs={"mask_type": "gaussian", "D0": D0},
    r=r,
    return_intermediates=True,
)

print("Intermediates keys:", list(inter.keys()))
print("Hb min/max:", inter["Hb"].min(), inter["Hb"].max())
print("G_shifted dtype:", inter["G_shifted"].dtype)
print("max imag in G_shifted:", np.max(np.abs(np.imag(inter["G_shifted"]))))
print("Done.")

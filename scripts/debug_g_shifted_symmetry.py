# scripts/debug_g_shifted_symmetry.py
import numpy as np
from io_utils.image_handler import read_image
from core.fft_engine import compute_fft, fft_shift
from core.color_processing import process_grayscale
from core import filters
from core.highboost import compose_highboost_mask

IMG = "data/sample1.png"   # change if you used another image
D0 = 10.0
r = 1.8

arr, meta = read_image(IMG)
if arr.ndim == 3:
    img = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
else:
    img = arr

F = compute_fft(img.astype(np.float64))
Fs = fft_shift(F)

# Build L and Hb exactly as pipeline does
L = filters.build_lowpass_mask(shape=Fs.shape, D0=D0, mask_type="gaussian", align_with_fftshift=True)
Hb = compose_highboost_mask(L, r)

G_shifted = Fs * Hb

# helper: correct Hermitian mapping
def hermitian_check_correct(A):
    M, N = A.shape
    i_idx = (-np.arange(M)) % M
    j_idx = (-np.arange(N)) % N
    B = np.conj(A[i_idx[:, None], j_idx[None, :]])
    diff = A - B
    max_abs = np.max(np.abs(diff))
    idx = np.unravel_index(np.argmax(np.abs(diff)), A.shape)
    return max_abs, idx, A[idx], B[idx], diff[idx]

print("Hb dtype:", Hb.dtype, "isrealobj(Hb)?", np.isrealobj(Hb))
print("Hb symmetric (flip both axes)?", np.allclose(Hb, Hb[::-1, ::-1]))
print("Fs dtype:", Fs.dtype, "iscomplexobj(Fs)?", np.iscomplexobj(Fs))

mF, idxF, AF, BF, dF = hermitian_check_correct(Fs)
mG, idxG, AG, BG, dG = hermitian_check_correct(G_shifted)
print(f"Fs Hermitian max diff: {mF}, idx={idxF}, Fs[idx]={AF}, conj_target[idx]={BF}")
print(f"G_shifted Hermitian max diff: {mG}, idx={idxG}, G[idx]={AG}, conj_target[idx]={BG}")
print("G_shifted[idx] - conj_target[idx] = ", dG)

# show small neighborhood at the worst G mismatch
i0, j0 = idxG
r = 2
print("Neighborhood (G_shifted.real):")
print(np.round(G_shifted.real[i0-r:i0+r+1, j0-r:j0+r+1], 6))
print("Neighborhood (G_shifted.imag):")
print(np.round(G_shifted.imag[i0-r:i0+r+1, j0-r:j0+r+1], 6))

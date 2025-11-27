import numpy as np
from core.highboost import compose_highboost_mask
from core.filters import build_lowpass_mask
from core.fft_engine import compute_fft, fft_shift

def test_Hb_monotonicity():
    L = np.random.rand(64,64)
    Hb1 = compose_highboost_mask(L, r=1.2)
    Hb5 = compose_highboost_mask(L, r=5.0)
    assert np.all(Hb5 >= Hb1)

def test_fft_hermitian_symmetry():
    img = np.random.rand(32,32)
    F = compute_fft(img)
    Fs = fft_shift(F)
    M,N = Fs.shape
    i_idx = (-np.arange(M)) % M
    j_idx = (-np.arange(N)) % N
    conj_pair = np.conj(Fs[i_idx[:,None], j_idx[None,:]])
    assert np.allclose(Fs, conj_pair, atol=1e-6)

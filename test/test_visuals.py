import os
import numpy as np
from visuals.plots import plot_mask, plot_highboost_mask, plot_magnitude_spectrum, compare_and_save

def _make_temp_dir(tmp_path, name="vis_tmp"):
    p = tmp_path / name
    p.mkdir()
    return str(p)

def test_plot_mask_and_highboost(tmp_path):
    outdir = _make_temp_dir(tmp_path)
    # simple circular mask
    shape = (32, 24)
    cx, cy = ( (shape[0]-1)/2.0, (shape[1]-1)/2.0 )
    # synthetic mask
    L = np.zeros(shape, dtype=float)
    L[shape[0]//2, shape[1]//2] = 1.0
    p1 = os.path.join(outdir, "mask.png")
    p2 = os.path.join(outdir, "hb.png")
    assert plot_mask(L, out_path=p1) == p1
    Hb = 1.5 + (1-1.5)*L
    assert plot_highboost_mask(Hb, out_path=p2) == p2
    assert os.path.exists(p1)
    assert os.path.exists(p2)

def test_plot_spectrum_and_compare(tmp_path):
    outdir = str(tmp_path / "spec")
    os.makedirs(outdir, exist_ok=True)
    # synthetic FFT: impulse at center (shifted)
    F = np.zeros((32,32), dtype=complex)
    F[16,16] = 1.0 + 0j
    p_spec = os.path.join(outdir, "spec.png")
    assert plot_magnitude_spectrum(F, out_path=p_spec) == p_spec
    # compare_and_save with small images
    orig = np.zeros((32,32), dtype=np.uint8)
    boosted = np.ones((32,32), dtype=np.uint8)*10
    pm = os.path.join(outdir, "cmp.png")
    assert compare_and_save(orig, boosted, spectrum=F, mask=None, Hb=None, out_path=pm) == pm
    assert os.path.exists(pm)

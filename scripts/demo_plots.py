"""
A small demo script that runs the core pipeline on a sample image (if present)
and writes a few visualization outputs into results/demo_plots/.
Run from project root:
python scripts/demo_plots.py
"""

import os
import numpy as np
from core.fft_engine import compute_fft, fft_shift
from core.color_processing import process_grayscale
from core import filters
from core.highboost import compose_highboost_from_mask_params
from visuals.plots import plot_magnitude_spectrum, plot_mask, plot_highboost_mask, compare_and_save
from io_utils.image_handler import read_image, save_image

OUTDIR = "results/demo_plots"

def demo_grayscale_from_file(input_path: str, D0: float = 10.0, r: float = 1.8):
    os.makedirs(OUTDIR, exist_ok=True)
    arr, meta = read_image(input_path)
    # convert to grayscale if color
    if arr.ndim == 3:
        # simple luminance approximation
        img = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    else:
        img = arr

    # run core pipeline manually to collect intermediates
    # compute FFT and shifted FFT
    F = compute_fft(img.astype(np.float64))
    Fs = fft_shift(F)

    # build masks
    L = filters.build_lowpass_mask(shape=Fs.shape, D0=D0, mask_type="gaussian", align_with_fftshift=True)
    Hb = compose_highboost_from_mask_params(shape=Fs.shape, D0=D0, mask_type="gaussian", r=r)

    # apply and get result via color_processing convenience (grayscale)
    out, intermediates = process_grayscale(img, mask_builder=None, mask_kwargs={"mask_type":"gaussian","D0":D0}, r=r, do_srgb_linearize=False, return_intermediates=True)
    # intermediates contains per-channel dict; for grayscale it's a single dict
    inter = intermediates

    # save plots
    plot_magnitude_spectrum(Fs, out_path=os.path.join(OUTDIR, "fft_spectrum.png"), is_shifted=True)
    plot_mask(L, out_path=os.path.join(OUTDIR, "filter_mask.png"))
    plot_highboost_mask(Hb, out_path=os.path.join(OUTDIR, "highpass_mask.png"))
    compare_and_save(img, out, spectrum=Fs, mask=L, Hb=Hb, out_path=os.path.join(OUTDIR, "comparison.png"))

    # save boosted image
    save_image(os.path.join(OUTDIR, "boosted_output.png"), out)
    print("Demo outputs written to:", OUTDIR)

if __name__ == "__main__":
    # try typical sample names used earlier; pick first existing
    candidates = ["data/sample3_checkerboard.tif", "data/sample1.png", "data/sample2_color.jpg"]
    found = None
    for c in candidates:
        if os.path.exists(c):
            found = c
            break
    if found is None:
        print("No sample image found in data/. Place one of sample1.png, sample2_color.jpg, sample3_checkerboard.tif and re-run.")
    else:
        demo_grayscale_from_file(found, D0=10.0, r=5)

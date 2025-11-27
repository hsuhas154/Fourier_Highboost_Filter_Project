"""
Batch-run demo across multiple images.

Saves per-image outputs and a CSV log with diagnostics:
- input_path, is_color, out_boosted_path, fft_spectrum_path, mask_path, highboost_mask_path,
  max_imag_G_shifted, max_imag_out_float, real_output_used (bool)

Usage (from project root):
python -m scripts.batch_demo

Edit the IMAGES list below to point to your 8 files if needed.
"""

import os
import csv
import time
from datetime import datetime
import numpy as np

from io_utils.image_handler import read_image, save_image, detect_is_color
from core.color_processing import process_color_rgb, process_grayscale
from core.fft_engine import compute_fft, fft_shift
from core import filters
from core.highboost import compose_highboost_from_mask_params
from visuals.plots import plot_magnitude_spectrum, plot_phase_spectrum, plot_mask, plot_highboost_mask, compare_and_save

# CONFIG: list image paths (the data/ directory in the project) you want to test (edit as needed)
IMAGES = [
    "data/Checkerboard_1.tif",
    "data/Checkerboard_2.jpg"
]

# pipeline params (tweak to make changes stronger/weaker)
DEFAULT_D0 = 50.0
DEFAULT_R = 6.0
MASK_TYPE = "radial"
BUTTER_ORDER = 3

# Control behavior:
# If force_real_output=True then we will take np.real(out_float) before final cast/save
force_real_output = True

# Safety threshold suggestion: if max_imag_out_float <= imag_threshold we consider it "small"
# Use relative threshold (fraction of max pixel value) or absolute value.
imag_threshold_abs = 3.0   # you can change this; for uint8 images 3 ~= ~1% of 255
# For relative threshold - e.g., 0.01 means 1% of max pixel range:
imag_threshold_rel = 0.01

# Output directory for this run
timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
OUTDIR = os.path.join("results", f"batch_demo_{timestamp}")
os.makedirs(OUTDIR, exist_ok=True)

# CSV log path
csv_path = os.path.join(OUTDIR, "results.csv")
csv_fields = [
    "input_path", "is_color", "out_boosted_path", "fft_spectrum_path", "mask_path",
    "highboost_mask_path", "max_imag_G_shifted", "max_imag_out_float", "used_real_output",
    "D0", "r", "mask_type",
]

def process_one_image(img_path, D0=DEFAULT_D0, r=DEFAULT_R):
    arr, meta = read_image(img_path)
    is_color = detect_is_color(arr)
    base = os.path.splitext(os.path.basename(img_path))[0]
    run_dir = os.path.join(OUTDIR, base)
    os.makedirs(run_dir, exist_ok=True)

    # Build mask & Hb for diagnostics (using shifted FFT shape)
    if is_color:
        H, W = arr.shape[0], arr.shape[1]
    else:
        H, W = arr.shape[0], arr.shape[1]

    # For diagnostics build FFT from a representative channel (grayscale use arr, color use luminance)
    if is_color:
        # simple luminance for mask size only
        rep = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float64)
    else:
        rep = arr.astype(np.float64)

    F = compute_fft(rep)
    Fs = fft_shift(F)
    L = filters.build_lowpass_mask(shape=Fs.shape, D0=D0, mask_type=MASK_TYPE, order=BUTTER_ORDER, align_with_fftshift=True)
    Hb = compose_highboost_from_mask_params(shape=Fs.shape, D0=D0, mask_type=MASK_TYPE, order=BUTTER_ORDER, r=r)

    # Run pipeline and request intermediates
    if is_color:
        out_img, inter_all = process_color_rgb(arr, mask_builder=None, mask_kwargs={"mask_type":MASK_TYPE,"D0":D0,"order":BUTTER_ORDER}, r=r, do_srgb_linearize=False, return_intermediates=True)
        # inter_all is dict with 'R','G','B' keys
        # We'll compute max imag metrics across channels (worst-case)
        max_imag_G = 0.0
        max_imag_out = 0.0
        for ch in ["R","G","B"]:
            ch_inter = inter_all[ch]
            max_imag_G = max(max_imag_G, float(np.max(np.abs(np.imag(ch_inter["G_shifted"])))))
            # out_float may be real; if complex handle accordingly
            if np.iscomplexobj(ch_inter.get("out_float", 0)):
                max_imag_out = max(max_imag_out, float(np.max(np.abs(np.imag(ch_inter["out_float"])))))
    else:
        out_img, inter = process_grayscale(arr, mask_builder=None, mask_kwargs={"mask_type":MASK_TYPE,"D0":D0,"order":BUTTER_ORDER}, r=r, do_srgb_linearize=False, return_intermediates=True)
        max_imag_G = float(np.max(np.abs(np.imag(inter["G_shifted"]))))
        max_imag_out = float(np.max(np.abs(np.imag(inter.get("out_float", 0))))) if np.iscomplexobj(inter.get("out_float", 0)) else 0.0

    # Decide whether to force real output
    used_real = False
    # compute a relative threshold based on dtype range
    if np.issubdtype(arr.dtype, np.integer):
        max_pixel_val = float(np.iinfo(arr.dtype).max)
    else:
        max_pixel_val = float(np.nanmax(arr))
        if max_pixel_val == 0:
            max_pixel_val = 1.0

    rel_thresh_val = imag_threshold_rel * max_pixel_val
    absolute_thresh = imag_threshold_abs

    # If the imag part is larger than BOTH relative and absolute thresholds, we will still produce output but mark used_real=True
    if force_real_output and (max_imag_out > rel_thresh_val or max_imag_out > absolute_thresh):
        # we already received out_img as a numpy array returned by process_* which is casted to dtype,
        # but if out_img still came from code where imaginary remnants were taken real, this step is safe.
        # To be safe, enforce taking real before saving:
        out_img = np.real(out_img).astype(arr.dtype)
        used_real = True

    # Save outputs
    boosted_path = os.path.join(run_dir, f"{base}_boosted.png")
    save_image(boosted_path, out_img)

    # Save diagnostic visual files
    fft_path = os.path.join(run_dir, "fft_spectrum.png")
    phase_path = os.path.join(run_dir, "phase_spectrum.png")
    mask_path = os.path.join(run_dir, "filter_mask.png")
    hb_path = os.path.join(run_dir, "highpass_mask.png")

    # save visualizations (Fs may be complex)
    try:
        plot_magnitude_spectrum(Fs, out_path=fft_path, is_shifted=True)
    except Exception as e:
        print("Warning: failed saving spectrum:", e)
    try:
        plot_phase_spectrum(Fs, out_path=phase_path, is_shifted=True)
    except Exception as e:
        print("Warning: failed saving phase spectrum:", e)
    try:
        plot_mask(L, out_path=mask_path)
    except Exception as e:
        print("Warning: failed saving mask:", e)
    try:
        plot_highboost_mask(Hb, out_path=hb_path)
    except Exception as e:
        print("Warning: failed saving Hb:", e)

    # return record
    return {
        "input_path": img_path,
        "is_color": bool(is_color),
        "out_boosted_path": boosted_path,
        "fft_spectrum_path": fft_path,
        "mask_path": mask_path,
        "highboost_mask_path": hb_path,
        "max_imag_G_shifted": max_imag_G,
        "max_imag_out_float": max_imag_out,
        "used_real_output": used_real,
        "D0": D0,
        "r": r,
        "mask_type": MASK_TYPE,
    }


def main():
    # prepare CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=csv_fields)
        writer.writeheader()

        for img in IMAGES:
            if not os.path.exists(img):
                print("Skipping missing:", img)
                continue
            print("Processing:", img)
            rec = process_one_image(img, D0=DEFAULT_D0, r=DEFAULT_R)
            writer.writerow(rec)
            csvf.flush()
            print(" -> done. max_imag_out:", rec["max_imag_out_float"], "used_real:", rec["used_real_output"])

    print("Batch done. Results in:", OUTDIR, "CSV:", csv_path)


if __name__ == "__main__":
    main()
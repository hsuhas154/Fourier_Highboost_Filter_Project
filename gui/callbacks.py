import os
import numpy as np
from tkinter import filedialog, messagebox
from core.color_processing import process_color_rgb, process_grayscale
from io_utils.image_handler import read_image, detect_is_color, save_image
from visuals.plots import compare_and_save
from .utils import safe_clip_real, np_to_tkimage

# ----------------------
# Simple normalization helpers
# ----------------------

def normalize_to_uint8(arr: np.ndarray, clip_percentiles=(1, 99)) -> np.ndarray:
    """
    Normalize a numeric array to uint8 for display.
    Uses percentile stretch by default to reduce effect of outliers.
    """
    arrf = arr.astype(np.float32)
    arrf = np.nan_to_num(arrf, nan=0.0, posinf=np.nanmax(arrf), neginf=np.nanmin(arrf))
    p_low, p_high = clip_percentiles

    try:
        if p_low is not None and p_high is not None and 0 <= p_low < p_high <= 100:
            vmin = float(np.percentile(arrf, p_low))
            vmax = float(np.percentile(arrf, p_high))
        else:
            vmin = float(np.min(arrf))
            vmax = float(np.max(arrf))
    except Exception:
        vmin = float(np.min(arrf))
        vmax = float(np.max(arrf))

    if vmax <= vmin:
        out = np.clip(arrf - vmin, 0, 255)
        return out.astype(np.uint8)

    scaled = (arrf - vmin) / (vmax - vmin)
    scaled = (scaled * 255.0).clip(0, 255)
    return scaled.astype(np.uint8)


def normalize_to_float_for_processing(arr: np.ndarray, clip_percentiles=(1, 99)) -> np.ndarray:
    """
    Normalize numeric array to float64 scaled to the range 0..255 for processing.
    Uses percentile stretch but returns float64.
    """
    arrf = arr.astype(np.float64)
    arrf = np.nan_to_num(arrf, nan=0.0, posinf=np.nanmax(arrf), neginf=np.nanmin(arrf))
    p_low, p_high = clip_percentiles

    try:
        if p_low is not None and p_high is not None and 0 <= p_low < p_high <= 100:
            vmin = float(np.percentile(arrf, p_low))
            vmax = float(np.percentile(arrf, p_high))
        else:
            vmin = float(np.min(arrf))
            vmax = float(np.max(arrf))
    except Exception:
        vmin = float(np.min(arrf))
        vmax = float(np.max(arrf))

    if vmax <= vmin:
        out = np.clip(arrf - vmin, 0.0, 255.0)
        return out.astype(np.float64)

    scaled = (arrf - vmin) / (vmax - vmin)
    scaled = (scaled * 255.0)
    return scaled.astype(np.float64)


# ----------------------
# Logging helper
# ----------------------
def _safe_log(app, *args, **kwargs):
    """Try to write to app.log if present, otherwise print to stdout."""
    msg = " ".join(str(a) for a in args) if args else kwargs.get("msg", "")
    try:
        if hasattr(app, "log") and callable(getattr(app, "log")):
            app.log(msg)
        else:
            print(msg)
    except Exception:
        print(msg)


# ----------------------
# Callbacks (simplified, stable flow)
# ----------------------
def open_image_callback(app):
    """Handle file selection and preview update (simple 8-bit workflow)."""
    path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.tif *.bmp *.gif *.avif"), ("All Files", "*.*")]
    )
    if not path:
        return

    # read image using project IO util
    arr, _meta = read_image(path)

    # Keep the raw array available for later processing.
    app.raw_image = arr

    # Debug log for evidence
    try:
        _safe_log(app, f"Loaded (Verbose): {path} shape={getattr(arr,'shape',None)} dtype={getattr(arr,'dtype',None)} min={np.nanmin(arr)} max={np.nanmax(arr)} meta={_meta} \n")
    except Exception:
        _safe_log(app, f"Loaded (Verbose): {path} (metadata unavailable) \n")

    # For preview: if already uint8, use as-is; otherwise percentile-scale to uint8
    if isinstance(arr, np.ndarray) and arr.dtype == np.uint8:
        # If color (H,W,3) or grayscale (H,W) already in correct type, use directly
        preview_arr = arr
    else:
        # percentiler stretch to uint8 for preview
        try:
            if arr.ndim == 3 and arr.shape[2] >= 3:
                chans = []
                for ch in range(min(3, arr.shape[2])):
                    chans.append(normalize_to_uint8(arr[..., ch], clip_percentiles=(1, 99)))
                preview_arr = np.stack(chans, axis=-1)
            elif arr.ndim == 3 and arr.shape[2] == 2:
                # two-channel -> treat as gray + alpha/second channel: use first channel for preview
                preview_arr = normalize_to_uint8(arr[..., 0], clip_percentiles=(1, 99))
            else:
                # single band or other -> grayscale preview
                preview_arr = normalize_to_uint8(arr, clip_percentiles=(1, 99))
        except Exception:
            # fallback to a safe uint8 cast of clipped values
            try:
                preview_arr = np.clip(arr, 0, 255).astype(np.uint8)
            except Exception:
                # final fallback: create a tiny black image so GUI doesn't break
                preview_arr = np.zeros((100, 100), dtype=np.uint8)

    # Prepare float array for processing (preserve dynamics)
    arr_for_proc = normalize_to_float_for_processing(arr, clip_percentiles=(1, 99))
    is_color_proc = detect_is_color(arr)

    # set preview image and attributes for the GUI
    app.image_np = preview_arr
    app.is_color = detect_is_color(preview_arr)
    try:
        if hasattr(app, "image_path"):
            app.image_path.set(path)
    except Exception:
        pass

    # show preview (guard if method exists)
    try:
        app.preview_original(preview_arr)
    except Exception:
        pass

    _safe_log(app, f"Loaded: {os.path.basename(path)} ({'Color' if app.is_color else 'Grayscale'}) \n")


def run_highboost_callback(app):
    """Perform high-boost filtering on current image (uses raw_image if present)."""
    if not hasattr(app, "image_np") or app.image_np is None:
        messagebox.showwarning("No Image", "Please open an image first.")
        return

    # Parameters from UI (guarded access)
    mask_type = getattr(app, "filter_type", None)
    try:
        mask_type = mask_type.get().lower() if mask_type is not None else "gaussian"
    except Exception:
        mask_type = "gaussian"

    try:
        D0 = float(app.D0_val.get())
    except Exception:
        D0 = 50.0
    try:
        r = float(app.r_val.get())
    except Exception:
        r = 1.8
    try:
        order = int(app.order_val.get()) if mask_type == "butterworth" else 2
    except Exception:
        order = 2
    do_linear = bool(getattr(app, "srgb_linearize", False) and getattr(app, "srgb_linearize").get())

    _safe_log(app, f"Running High-Boost ({mask_type}, D0={D0}, r={r}, order={order}) ... \n")

    # Decide which array to process: prefer raw_image if available
    if hasattr(app, "raw_image") and app.raw_image is not None:
        arr_for_proc = normalize_to_float_for_processing(app.raw_image, clip_percentiles=(1, 99))
        is_color_proc = detect_is_color(app.raw_image)
    else:
        arr_for_proc = normalize_to_float_for_processing(app.image_np, clip_percentiles=(1, 99))
        is_color_proc = getattr(app, "is_color", False)

    is_color = bool(is_color_proc)

    try:
        if is_color:
            out_img = process_color_rgb(
                arr_for_proc, mask_builder=None,
                mask_kwargs={"mask_type": mask_type, "D0": D0, "order": order},
                r=r, do_srgb_linearize=do_linear, return_intermediates=False
            )
        else:
            out_img = process_grayscale(
                arr_for_proc, mask_builder=None,
                mask_kwargs={"mask_type": mask_type, "D0": D0, "order": order},
                r=r, do_srgb_linearize=False, return_intermediates=False
            )
    except Exception as e:
        _safe_log(app, f"Processing error: {e} \n")
        messagebox.showerror("Processing Error", f"High-boost processing failed:\n{e}")
        return

    out_img = safe_clip_real(out_img)
    app.output_np = out_img
    try:
        app.preview_boosted(out_img)
    except Exception:
        pass

    _safe_log(app, "High-Boost complete. \n")


def save_output_callback(app):
    """Save the last computed boosted image to a file (safe logging)."""
    try:
        if not hasattr(app, "output_np") or app.output_np is None:
            messagebox.showwarning("No Output", "There is no boosted image to save. Run the filter first.")
            return

        suggested = os.path.join("results", "boosted_output.png")
        save_path = filedialog.asksaveasfilename(
            title="Save Boosted Image",
            initialfile=os.path.basename(suggested),
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg;*.jpeg"), ("TIFF Image", "*.tif;*.tiff")]
        )
        if not save_path:
            _safe_log(app, "Save cancelled. \n")
            return

        try:
            from io_utils.image_handler import save_image
            save_image(save_path, app.output_np)
        except Exception:
            from PIL import Image
            arr = app.output_np
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(save_path)

        _safe_log(app, f"Saved boosted image → {save_path} \n")
        try:
            messagebox.showinfo("Saved", f"Saved boosted image →\n{save_path}")
        except Exception:
            pass

    except Exception as e:
        _safe_log(app, f"Save failed: {e} \n")
        try:
            messagebox.showerror("Save Error", f"Saving failed:\n{e}")
        except Exception:
            pass

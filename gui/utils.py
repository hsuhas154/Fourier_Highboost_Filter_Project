import numpy as np
from PIL import Image, ImageTk

def np_to_tkimage(arr):
    """Convert a numpy array (H×W or H×W×3) to a PhotoImage for tkinter."""
    if arr.ndim == 2:
        img = Image.fromarray(np.uint8(arr))
    else:
        img = Image.fromarray(np.uint8(arr[..., :3]))
    return ImageTk.PhotoImage(img)

def safe_clip_real(img):
    """Ensure real uint8 image output for display."""
    img = np.real(img)
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

import ttkbootstrap as ttk
import tkinter as tk
from ttkbootstrap.constants import *
from tkinter import StringVar, DoubleVar, IntVar, BooleanVar
from tkinter.scrolledtext import ScrolledText
from .callbacks import open_image_callback, run_highboost_callback, save_output_callback
from .utils import np_to_tkimage
import numpy as np
from PIL import Image, ImageTk

class HighBoostApp(ttk.Window):
    def __init__(self, title="Fourier High-Boost Filter", themename="cyborg"):
        super().__init__(themename=themename)
        self.title(title)
        self.geometry("1200x700")

        # Data
        self.image_np = None
        self.output_np = None
        self.is_color = False
        self.image_path = StringVar()

        # Variables
        self.filter_type = StringVar(value="gaussian")
        self.D0_val = DoubleVar(value=50.0)
        self.r_val = DoubleVar(value=1.8)
        self.order_val = IntVar(value=2)
        self.srgb_linearize = BooleanVar(value=False)

        # Build UI
        self._build_layout()


    def log(self, msg: str):
        """
        GUI logger: append message to the ScrolledText log_box if available,
        otherwise print to stdout. Keeps GUI from crashing if log_box isn't ready.
        """
        try:
            # prefer GUI log area if it exists
            if hasattr(self, "log_box") and self.log_box is not None:
                try:
                    self.log_box.insert("end", str(msg) + "\n")
                    self.log_box.see("end")
                    return
                except Exception:
                    # fallback to print if GUI widget fails
                    print(str(msg))
                    return
            else:
                # no GUI log widget created yet
                print(str(msg))
        except Exception:
            # very last resort
            print(str(msg))


    # --- Layout ---
    def _build_layout(self):
        # Left control panel
        control = ttk.Frame(self)
        control.pack(side=LEFT, fill=Y, padx=10, pady=10)

        # --- Filter type selection ---
        ttk.Label(control, text="Filter Type:").pack(anchor=W)
        opt = ttk.OptionMenu(control, self.filter_type, "Gaussian", "Gaussian", "Radial", "Butterworth")
        opt.pack(fill=X, pady=2)

        # Cutoff and r
        ttk.Label(control, text="Cutoff D₀:").pack(anchor=W)
        ttk.Entry(control, textvariable=self.D0_val).pack(fill=X, pady=2)

        # Boost factor r
        ttk.Label(control, text="Boost Factor r:").pack(anchor=W)
        ttk.Entry(control, textvariable=self.r_val).pack(fill=X, pady=2)

        # Create a placeholder frame for Butterworth controls, but DO NOT pack it now.
        # We will pack it dynamically before the sRGB checkbox when needed.
        self._butter_placeholder = ttk.Frame(control)
        # create the label + entry as children of the placeholder (do not pack them yet)
        self._butter_label = ttk.Label(self._butter_placeholder, text="Butterworth Order:")
        self.order_entry = ttk.Entry(self._butter_placeholder, textvariable=self.order_val)

        # Create and pack the sRGB checkbox now (we will pack placeholder BEFORE this widget when showing)
        self.srgb_cb = ttk.Checkbutton(control, text="sRGB Linearize", variable=self.srgb_linearize)
        self.srgb_cb.pack(anchor=W, pady=5)

        # Initialize visibility (will not show since placeholder is not packed)
        self._update_order_state()
        try:
            self.filter_type.trace_add("write", lambda *args: self._update_order_state())
        except AttributeError:
            self.filter_type.trace("w", lambda *args: self._update_order_state())

        # Buttons
        ttk.Button(control, text="Open Image", bootstyle=PRIMARY, command=lambda: open_image_callback(self)).pack(fill=X, pady=3)
        ttk.Button(control, text="Run High-Boost", bootstyle=SUCCESS, command=lambda: run_highboost_callback(self)).pack(fill=X, pady=3)
        ttk.Button(control, text="Save Output", bootstyle=INFO, command=lambda: save_output_callback(self)).pack(fill=X, pady=3)

        ttk.Separator(control).pack(fill=X, pady=5)
        ttk.Label(control, text="Logs:").pack(anchor=W)
        # use the standard tkinter scrolled text widget
        self.log_box = ScrolledText(control, height=15, wrap="word")
        self.log_box.configure(font=("Helvetica", 10))
        self.log_box.pack(fill=BOTH, expand=True, pady=5)

        # Right display area: two panes for original and boosted images, each with a scrollable canvas
        display = ttk.Frame(self)
        display.pack(side=LEFT, fill=BOTH, expand=True, padx=(8,12), pady=8)


        def _make_preview_column(parent):
            col = ttk.Frame(parent)
            # toolbar: zoom in / zoom out / fit
            toolrow = ttk.Frame(col)
            toolrow.pack(fill=X, anchor=N, pady=(0,4))
            btn_zoom_in = ttk.Button(toolrow, text="+", width=3, command=lambda: self._zoom(1.2))
            btn_zoom_out = ttk.Button(toolrow, text="-", width=3, command=lambda: self._zoom(1/1.2))
            btn_fit = ttk.Button(toolrow, text="Fit", width=6, command=lambda: self._fit_image())
            btn_zoom_in.pack(side=LEFT, padx=2)
            btn_zoom_out.pack(side=LEFT, padx=2)
            btn_fit.pack(side=LEFT, padx=6)

            # canvas + scrollbars
            canvas_frame = ttk.Frame(col)
            canvas_frame.pack(fill=BOTH, expand=True)
            canvas = tk.Canvas(canvas_frame, background="black")
            hbar = ttk.Scrollbar(canvas_frame, orient="horizontal", command=canvas.xview)
            vbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
            canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

            hbar.pack(side=BOTTOM, fill=X)
            vbar.pack(side=RIGHT, fill=Y)
            canvas.pack(side=LEFT, fill=BOTH, expand=True)

            return col, canvas

        # create left and right preview columns
        left_col, self.orig_canvas = _make_preview_column(display)
        left_col.pack(side=LEFT, fill=BOTH, expand=True, padx=(0,6))
        right_col, self.boost_canvas = _make_preview_column(display)
        right_col.pack(side=LEFT, fill=BOTH, expand=True)

        # initialize canvas state
        self._orig_img_id = None
        self._boost_img_id = None
        self._orig_tkimage = None
        self._boost_tkimage = None
        self._zoom_factor = 1.0
        self._current_canvas = None

        # bind mouse wheel scroll / zoom to both canvases
        for c in (self.orig_canvas, self.boost_canvas):
            # Windows / Mac / Linux wheel bindings
            c.bind("<Enter>", lambda e, canvas=c: self._set_current_canvas(canvas))
            c.bind("<Leave>", lambda e: self._set_current_canvas(None))
            c.bind("<Button-4>", lambda e: self._on_mousewheel(e, 1.0))   # some X11
            c.bind("<Button-5>", lambda e: self._on_mousewheel(e, -1.0))
            c.bind("<MouseWheel>", lambda e: self._on_mousewheel(e, None))


    def _update_order_state(self):
        """
        Show/hide Butterworth order controls.
        When showing, pack the placeholder BEFORE the sRGB checkbox so it sits right under Boost Factor.
        When hiding, forget both the children and the placeholder so no gap remains.
        """
        try:
            current = str(self.filter_type.get()).strip().lower()
        except Exception:
            current = ""

        if current == "butterworth":
            # If placeholder not packed, pack it right before the sRGB checkbox so ordering is correct
            if not getattr(self._butter_placeholder, "_is_packed", False):
                # pack placeholder before the sRGB checkbox
                # use fill=X so children can use full width
                self._butter_placeholder.pack(fill=X, before=self.srgb_cb)
                self._butter_placeholder._is_packed = True

            # pack children inside placeholder if not already packed
            if not getattr(self.order_entry, "_is_packed", False):
                self._butter_label.pack(anchor=W, side=TOP, fill=X, pady=(4, 0))
                self.order_entry.pack(fill=X, pady=(2, 6))
                self.order_entry._is_packed = True

            # enable editing
            self.order_entry.configure(state="normal")
        else:
            # hide the label+entry inside placeholder if they were packed
            if getattr(self.order_entry, "_is_packed", False):
                self._butter_label.pack_forget()
                self.order_entry.pack_forget()
                self.order_entry._is_packed = False

            # if placeholder itself was packed, forget it so no empty gap remains
            if getattr(self._butter_placeholder, "_is_packed", False):
                self._butter_placeholder.pack_forget()
                self._butter_placeholder._is_packed = False

            # keep entry disabled when hidden
            self.order_entry.configure(state="disabled")

    def _set_current_canvas(self, canvas):
        self._current_canvas = canvas


    def _on_mousewheel(self, event, delta_override):
        """
        Mouse wheel handler:
        - If Ctrl is held, zoom around pointer
        - Otherwise perform vertical scroll on current canvas
        """
        canvas = self._current_canvas
        if canvas is None:
            return

        # Determine wheel delta in a cross-platform way
        if delta_override is not None:
            delta = delta_override
        else:
            # Windows: event.delta is multiple of 120; Mac similar; normalise
            delta = event.delta / 120.0

        # If Ctrl is held -> zoom
        ctrl = (event.state & 0x4) != 0
        if ctrl:
            factor = 1.2 if delta > 0 else (1.0 / 1.2)
            self._zoom(factor, canvas=canvas, center=(event.x, event.y))
        else:
            # scroll vertically
            canvas.yview_scroll(int(-1 * delta), "units")


    def _zoom(self, factor, canvas=None, center=None):
        """
        Zoom the image displayed on both canvases (if present) by 'factor'.
        If canvas parameter specified, we zoom that canvas and center around (x,y) pixel.
        """
        self._zoom_factor *= factor
        # clamp zoom
        self._zoom_factor = max(0.1, min(self._zoom_factor, 10.0))

        # helper to resize and redraw on a canvas
        def _redraw_on(canvas, tkimage_attr, img_id_attr, orig_np):
            if orig_np is None:
                return
            # compute new size
            h, w = orig_np.shape[:2]
            new_w = int(w * self._zoom_factor)
            new_h = int(h * self._zoom_factor)
            # use PIL to resize (fast nearest or bilinear)
            pil = Image.fromarray(orig_np.astype(np.uint8))
            pil_resized = pil.resize((new_w, new_h), Image.BILINEAR)
            tkimg = ImageTk.PhotoImage(pil_resized)
            setattr(self, tkimage_attr, tkimg)  # keep reference
            # clear previous
            prev = getattr(self, img_id_attr)
            canvas.delete("all")
            # create image at top-left and set scrollregion
            imgid = canvas.create_image(0, 0, anchor="nw", image=tkimg)
            canvas.config(scrollregion=canvas.bbox("all"))
            setattr(self, img_id_attr, imgid)

        # If canvas param provided, only update that one; else update both if they exist
        if canvas is None or canvas == self.orig_canvas:
            _redraw_on(self.orig_canvas, "_orig_tkimage", "_orig_img_id", getattr(self, "_orig_np", None))
        if canvas is None or canvas == self.boost_canvas:
            _redraw_on(self.boost_canvas, "_boost_tkimage", "_boost_img_id", getattr(self, "_boost_np", None))


    def _fit_image(self):
        """
        Fit the current image(s) to the canvas size (sets zoom_factor accordingly).
        When multiple images present we fit each independently to its canvas.
        """
        def _fit(canvas, orig_np, tkimage_attr, img_id_attr):
            if orig_np is None:
                return
            cw = canvas.winfo_width()
            ch = canvas.winfo_height()
            if cw <= 1 or ch <= 1:
                return
            h, w = orig_np.shape[:2]
            scale = min(cw / w, ch / h, 1.0)
            self._zoom_factor = scale
            # Redraw using _zoom with specific factor computed (we override)
            # create resized image directly
            new_w = int(w * self._zoom_factor)
            new_h = int(h * self._zoom_factor)
            pil = Image.fromarray(orig_np.astype(np.uint8))
            pil_resized = pil.resize((new_w, new_h), Image.BILINEAR)
            tkimg = ImageTk.PhotoImage(pil_resized)
            setattr(self, tkimage_attr, tkimg)
            canvas.delete("all")
            imgid = canvas.create_image(0, 0, anchor="nw", image=tkimg)
            canvas.config(scrollregion=canvas.bbox("all"))
            setattr(self, img_id_attr, imgid)

        _fit(self.orig_canvas, getattr(self, "_orig_np", None), "_orig_tkimage", "_orig_img_id")
        _fit(self.boost_canvas, getattr(self, "_boost_np", None), "_boost_tkimage", "_boost_img_id")


    def preview_original(self, arr):
        # keep original numpy array
        self._orig_np = arr.copy()
        self._zoom_factor = 1.0
        # draw to canvas at original size
        pil = Image.fromarray(arr.astype(np.uint8))
        tkimg = ImageTk.PhotoImage(pil)
        self._orig_tkimage = tkimg
        self.orig_canvas.delete("all")
        self._orig_img_id = self.orig_canvas.create_image(0, 0, anchor="nw", image=tkimg)
        self.orig_canvas.config(scrollregion=self.orig_canvas.bbox("all"))


    def preview_boosted(self, arr):
        self._boost_np = arr.copy()
        self._zoom_factor = 1.0
        pil = Image.fromarray(arr.astype(np.uint8))
        tkimg = ImageTk.PhotoImage(pil)
        self._boost_tkimage = tkimg
        self.boost_canvas.delete("all")
        self._boost_img_id = self.boost_canvas.create_image(0, 0, anchor="nw", image=tkimg)
        self.boost_canvas.config(scrollregion=self.boost_canvas.bbox("all"))


def launch_app():
    app = HighBoostApp()
    app.mainloop()

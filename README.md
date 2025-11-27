# Fourier-Domain High-Boost Image Filter

### Course: GNR607 - Satellite Image Processing  
### Institution: Indian Institute of Technology Bombay  
### Authors: Srinija Enimireddy, Atharva Wakulkar, Suhas H

---

## 1. Project Overview

This project implements a **High-Boost Filtering tool in the Frequency Domain**.  
Unlike spatial domain sharpening (which uses small convolution masks), this application transforms images into the frequency domain using the **Fast Fourier Transform (FFT)**. It allows precise manipulation of frequency components to amplify details (edges) while preserving the low-frequency base.

The application features a modern **GUI built with ttkbootstrap** for interactive parameter tuning and real-time visualization of the original vs. boosted images.

---

## Key Features

### **Frequency Domain Engine**  
Implements 2D FFT and IFFT with proper center shifting logic.

### **Custom Filter Kernels**
- **Ideal (Radial):** Sharp frequency cutoff  
- **Gaussian:** Smooth attenuation (no ringing artifacts)  
- **Butterworth:** Controlled roll-off via order parameter (\(n\))

### **High-Boost Logic**  
Implements the formula: 
$$
H_b(u,v) = r + (1 - r)L(u,v)
$$
to sharpen images.

### **Color Fidelity**  
Processes R, G, and B channels independently to maintain color accuracy.

### **Diagnostics**  
Includes scripts to verify Hermitian symmetry and ensure output images remain real-valued.

---

## 2. Directory Structure

    FOURIER_HIGHBOOST_FILTER_PROJECT/
    ├── main.py                     # Entry point for the GUI application
    ├── pytest.ini                  # Configuration for test suite
    ├── requirements.txt            # Python dependencies
    ├── README.md                   # Project description and usage guide
    ├── core/                       # Mathematical Core
    │   ├── fft_engine.py           # Wrappers for FFT/IFFT and shift operations
    │   ├── filters.py              # Generation of Gaussian/Butterworth/Radial masks
    │   ├── highboost.py            # High-boost formula logic
    │   ├── color_processing.py     # RGB channel splitting and recombination
    │   └── __init__.py
    ├── gui/                        # User Interface
    │   ├── interface.py            # Main Window layout (ttkbootstrap)
    │   ├── callbacks.py            # Event handling (Load, Run, Save)
    │   ├── utils.py                # GUI helper functions
    │   └── __init__.py
    ├── io_utils/                   # File Input/Output
    │   ├── image_handler.py        # Pillow-based reading/writing (RGB/Grayscale)
    │   ├── file_utils.py           # Timestamped filename generation
    │   └── __init__.py
    ├── visuals/                    # Visualization Tools
    │   ├── plots.py                # Matplotlib utilities for spectra and masks
    │   └── __init__.py
    ├── scripts/                    # Debugging & Demo Tools
    │   ├── batch_demo.py           # Batch processing script with CSV reporting
    │   ├── demo_plots.py           # Script to generate documentation plots
    │   ├── debug_intermediates.py  # Diagnostic tool for pipeline steps
    │   └── debug_g_shifted_symmetry.py # Verification of Hermitian symmetry
    ├── tests/                      # Unit Testing Suite
    │   ├── test_fft_engine.py
    │   ├── test_filters.py
    │   ├── test_highboost.py
    │   ├── test_color_processing.py
    │   ├── test_io_utils.py
    │   ├── test_invariants.py
    │   ├── test_visuals.py
    │   ├── test_pipeline_large_image.py
    │   └── test_pipeline_basic.py
    ├── data/                       # This folder is used to include some sample images
    ├── docs/                       # This folder is used to include all the documentation files
    └── results/                    # This folder displays batch processing results generated via batch_demo.py which contains magnitude spectrum, masks, highboost output and CSV logs

---

## 3. Installation

### Option 1

#### Prerequisites
- Python **3.10 or newer**
- A virtual environment is recommended

### Setup
- Navigate to the project directory:

```bash
    cd FOURIER_HIGHBOOST_FILTER_PROJECT
```

- Install required packages:

```bash
pip install -r requirements.txt
```

- Dependencies include: numpy, matplotlib, Pillow, ttkbootstrap, pytest.

#### Option 2: Using Conda
To create an exact replica of the development environment:

```bash
conda env create -f environment.yml
```

**Note: This would install everything in the base environment. If you want to create a new project specific environment and installl all the required packages and libraries in that enviroment, which is highly recommended, the cooresponding instructions are given in the User_Manual PDF present in the data/ directory.**

---

## 4. How to Run

### Method 1: Graphical User Interface (GUI)
- This is the primary mode of operation.

```bash
python main.py
```

- Open Image: Click the blue Open Image button to load a file (JPG, PNG, TIF).

### Adjust Parameters:
- Filter Type: Select Gaussian, Butterworth, or Radial.
- Cutoff ($D_0$): Radius of the low-pass mask (e.g., 50.0).
- Boost Factor ($r$): Strength of sharpening (Must be $> 1.0$).
- Order: (For Butterworth) Steepness of the filter curve.
- Execute: Click Run High-Boost. The processed image appears in the right pane.
- Controls: Use the mouse wheel to zoom in/out of the images.
- Export: Click Save Output to save the result to the results/ folder.

### Method 2: Batch Processing
**Caution: This is for testing only.**
- To process a folder of images automatically and generate diagnostic reports (CSV + Plots):

```bash
python -m scripts.batch_demo
```

- This will create a timestamped folder in results/ containing boosted images, magnitude spectra, and a results.csv summary.

---

## 5. Theoretical Background

- The high-boost filtering process amplifies high-frequency components (edges) while keeping the low-frequency components (background) intact.
- Transform: The input image $f(x,y)$ is transformed to the frequency domain $F(u,v)$.
- Filter Design: A low-pass filter $L(u,v)$ is created (e.g., Gaussian).
- High-Boost Mask: The mask is inverted and scaled:
$$
H_b(u,v) = r + (1-r)L(u,v)
$$
- If $r=1$, it returns the original image.
- If $r>1$, high frequencies are boosted.
- Apply & Invert: The mask is multiplied with the spectrum ($G = F \cdot H_b$), and the Inverse FFT generates the sharpened spatial image.

---

6. Testing & Validation

The project uses pytest to ensure mathematical accuracy. The suite checks for:
- Correctness of FFT/IFFT round-trips.
- Hermitian symmetry in frequency masks (crucial for real-valued output).
- Channel independence in RGB processing.
- To run the tests:
```bash
pytest -v
```

---

## 7. Acknowledgements

- Course Instructor: Prof. B Krishna Mohan for the theoretical foundation.
- Open Source: Built using the Python scientific stack (NumPy, SciPy) and ttkbootstrap for the UI.

---
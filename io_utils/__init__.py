# io/__init__.py
"""
I/O helpers package for Fourier High-Boost project.
"""
from .image_handler import read_image, save_image, detect_is_color
from .file_utils import make_result_filename, save_parameters_txt, zip_results

__all__ = [
    "read_image",
    "save_image",
    "detect_is_color",
    "make_result_filename",
    "save_parameters_txt",
    "zip_results",
]

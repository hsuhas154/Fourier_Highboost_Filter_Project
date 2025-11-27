# visuals/__init__.py
"""
Visual helpers for the Fourier High-Boost project.
Provides plotting and export utilities used by GUI/CLI.
"""
from .plots import (
    plot_magnitude_spectrum,
    plot_mask,
    plot_highboost_mask,
    compare_and_save,
    plot_channel_breakdown,
    fig_to_array,
)
__all__ = [
    "plot_magnitude_spectrum",
    "plot_mask",
    "plot_highboost_mask",
    "compare_and_save",
    "plot_channel_breakdown",
    "fig_to_array",
]

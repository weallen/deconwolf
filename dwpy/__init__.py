"""
dwpy - DeconWolf Python Package

Fast, flexible deconvolution backends for microscopy image processing.
"""

__version__ = "0.1.0"

# Core deconvolution functions
from .dw_numpy import DeconvolutionConfig, deconvolve, deconvolve_tiled

# Fast backend dispatcher
from .dw_fast import deconvolve_fast

# Auto-configuration
from .dw_auto import (
    auto_deconvolve,
    auto_config,
    recommend_backend,
    estimate_memory_usage,
)

# PSF generation
from .psf import generate_psf_bw, generate_psf_gl

# Optional backends
try:
    from .dw_dask import dask_deconvolve
except ImportError:
    dask_deconvolve = None

try:
    from .dw_jax import deconvolve_jax
except ImportError:
    deconvolve_jax = None

__all__ = [
    # Core API
    "DeconvolutionConfig",
    "deconvolve",
    "deconvolve_tiled",
    "deconvolve_fast",
    # Auto-config
    "auto_deconvolve",
    "auto_config",
    "recommend_backend",
    "estimate_memory_usage",
    # PSF
    "generate_psf_bw",
    "generate_psf_gl",
    # Optional
    "dask_deconvolve",
    "deconvolve_jax",
    # Version
    "__version__",
]

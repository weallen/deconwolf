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
from .psf_utils import (
    auto_psf_size,
    auto_psf_size_physical,
    auto_psf_size_c_heuristic,
    calculate_psf_size,
    pad_psf_to_image_size,
    auto_generate_psf_bw,
    auto_generate_psf_gl,
    explain_tiled_deconvolution,
)

# Configuration system
from .config_schema import (
    MicroscopeConfig,
    ImagingConfig,
    PSFConfig,
    DeconvolutionConfigParams,
    ExperimentConfig,
)
from .config_loader import load_experiment_config, save_experiment_config
from .workflow import generate_psf_from_config, deconvolve_from_config

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
    # PSF generation
    "generate_psf_bw",
    "generate_psf_gl",
    # PSF utilities
    "auto_psf_size",
    "auto_psf_size_physical",
    "auto_psf_size_c_heuristic",
    "calculate_psf_size",
    "pad_psf_to_image_size",
    "auto_generate_psf_bw",
    "auto_generate_psf_gl",
    "explain_tiled_deconvolution",
    # Configuration system
    "MicroscopeConfig",
    "ImagingConfig",
    "PSFConfig",
    "DeconvolutionConfigParams",
    "ExperimentConfig",
    "load_experiment_config",
    "save_experiment_config",
    "generate_psf_from_config",
    "deconvolve_from_config",
    # Optional
    "dask_deconvolve",
    "deconvolve_jax",
    # Version
    "__version__",
]

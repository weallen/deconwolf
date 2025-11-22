"""
High-level workflow functions for experiment-based deconvolution.

Provides simplified APIs that combine PSF generation, configuration, and
deconvolution into single function calls.
"""

import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, Tuple, Optional

from .config_loader import load_experiment_config
from .config_schema import ExperimentConfig
from .psf_utils import auto_psf_size_c_heuristic
from .psf import generate_psf_bw, generate_psf_gl
from .dw_numpy import DeconvolutionConfig
from .dw_fast import deconvolve_fast


def generate_psf_from_config(
    config: ExperimentConfig,
    image_shape: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """
    Generate PSF from experiment configuration.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration
    image_shape : tuple, optional
        Image shape (X, Y, Z) for reference (not used for sizing currently)

    Returns
    -------
    np.ndarray
        Generated PSF, normalized to sum=1.0

    Examples
    --------
    >>> config = load_experiment_config('configs/dapi_60x_oil.yaml')
    >>> psf = generate_psf_from_config(config)
    >>> print(psf.shape)
    (181, 181, 183)
    """
    # Calculate PSF size using C heuristic
    xy_size, z_size = auto_psf_size_c_heuristic(
        dxy=config.imaging.dxy,
        dz=config.imaging.dz,
        NA=config.microscope.NA,
        wvl=config.imaging.wavelength,
        ni=config.microscope.ni,
        xy_size=config.psf.xy_size,  # None = auto
        z_size=config.psf.z_size,
    )

    # Auto-select PSF model if needed
    model = config.psf.model.lower()
    if model == "auto":
        # Select based on RI mismatch
        if config.microscope.ns is None:
            model = "bw"  # No specimen RI specified
        elif abs(config.microscope.ns - config.microscope.ni) < 0.05:
            model = "bw"  # RI matched (< 5% difference)
        else:
            model = "gl"  # RI mismatch, use Gibson-Lanni

    # Generate PSF
    if model in ["gl", "gibson-lanni"]:
        if config.microscope.ns is None:
            raise ValueError(
                "Gibson-Lanni PSF requires specimen RI (microscope.ns). "
                "Set ns in config or use model='bw'"
            )

        psf = generate_psf_gl(
            dxy=config.imaging.dxy,
            dz=config.imaging.dz,
            xy_size=xy_size,
            z_size=z_size,
            NA=config.microscope.NA,
            ni=config.microscope.ni,
            ns=config.microscope.ns,
            wvl=config.imaging.wavelength,
            M=config.microscope.M,
            ti0=config.microscope.ti0,
            tg=config.microscope.tg,
            ng=config.microscope.ng,
            ni0=config.psf.ni0,
            tg0=config.psf.tg0,
            ng0=config.psf.ng0,
        )
    else:  # "bw", "born-wolf"
        psf = generate_psf_bw(
            dxy=config.imaging.dxy,
            dz=config.imaging.dz,
            xy_size=xy_size,
            z_size=z_size,
            NA=config.microscope.NA,
            ni=config.microscope.ni,
            wvl=config.imaging.wavelength,
        )

    return psf


def deconvolve_from_config(
    image: np.ndarray,
    config: Union[str, Path, ExperimentConfig],
    psf: Optional[np.ndarray] = None,
    **overrides
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Complete deconvolution workflow from configuration.

    Parameters
    ----------
    image : np.ndarray
        Input image (X, Y, Z)
    config : str, Path, or ExperimentConfig
        Path to config file or loaded config object
    psf : np.ndarray, optional
        Pre-computed PSF. If None, generates from config
    **overrides
        Override config parameters (e.g., n_iter=100, backend='numba')

    Returns
    -------
    result : np.ndarray
        Deconvolved image
    info : dict
        Information about the run (psf, config, timing, etc.)

    Examples
    --------
    >>> # Simple usage
    >>> result, info = deconvolve_from_config(
    ...     image, 'configs/dapi_60x_oil.yaml'
    ... )

    >>> # With overrides
    >>> result, info = deconvolve_from_config(
    ...     image, 'configs/dapi_60x_oil.yaml',
    ...     n_iter=100, backend='numba'
    ... )

    >>> # With pre-computed PSF
    >>> psf = generate_psf_from_config(config)
    >>> result, info = deconvolve_from_config(image, config, psf=psf)
    """
    import time

    # Load config if needed
    if isinstance(config, (str, Path)):
        config = load_experiment_config(config)

    # Apply overrides to deconvolution parameters
    deconv_params = vars(config.deconvolution).copy()
    deconv_params.update(overrides)

    # Extract method and backend (not part of DeconvolutionConfig)
    method = deconv_params.pop('method', 'shb')
    backend = deconv_params.pop('backend', 'jax')

    # Create DeconvolutionConfig
    cfg = DeconvolutionConfig(**deconv_params)

    # Generate PSF if not provided
    if psf is None:
        print(f"Generating PSF from config: {config.name}")
        psf = generate_psf_from_config(config, image.shape)
        print(f"  PSF: {psf.shape}, model={config.psf.model}")

    # Run deconvolution
    print(f"Running deconvolution:")
    print(f"  Method: {method}, Backend: {backend}")
    print(f"  Iterations: {cfg.n_iter}")

    start = time.perf_counter()
    result = deconvolve_fast(
        image, psf,
        method=method,
        backend=backend,
        cfg=cfg
    )
    elapsed = time.perf_counter() - start

    print(f"✓ Completed in {elapsed:.2f}s ({elapsed/cfg.n_iter:.2f}s/iter)")

    # Return result and info
    info = {
        'psf': psf,
        'config': config,
        'time': elapsed,
        'method': method,
        'backend': backend,
        'iterations': cfg.n_iter,
    }

    return result, info


__all__ = [
    'generate_psf_from_config',
    'deconvolve_from_config',
]

"""
Automatic parameter selection for deconvolution based on image/PSF properties.
"""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from .dw_numpy import DeconvolutionConfig


def estimate_memory_usage(
    im_shape: Tuple[int, int, int],
    psf_shape: Tuple[int, int, int],
    border_quality: int = 2,
    n_iter: int = 20,
) -> float:
    """
    Estimate peak memory usage in GB for deconvolution.

    Memory is dominated by:
    - Work array (padded image): wM × wN × wP
    - FFT arrays (complex): ~2x work array
    - Temporary arrays: ~3x work array
    Total: ~6x work array size
    """
    M, N, P = im_shape
    pM, pN, pP = psf_shape

    if border_quality == 0:
        wM, wN, wP = max(M, pM), max(N, pN), max(P, pP)
    elif border_quality == 1:
        wM = M + (pM + 1) // 2
        wN = N + (pN + 1) // 2
        wP = P + (pP + 1) // 2
    else:  # border_quality == 2
        wM = M + pM - 1
        wN = N + pN - 1
        wP = P + pP - 1

    bytes_per_float = 4
    work_size = wM * wN * wP * bytes_per_float
    total_bytes = work_size * 6  # Account for FFTs and temporaries
    return total_bytes / (1024**3)  # Convert to GB


def auto_config(
    im: np.ndarray,
    psf: np.ndarray,
    quality: str = "balanced",
    max_memory_gb: Optional[float] = None,
) -> DeconvolutionConfig:
    """
    Automatically determine optimal deconvolution parameters.

    Parameters
    ----------
    im : np.ndarray
        Input image (M, N, P) in XYZ order
    psf : np.ndarray
        Point spread function (pM, pN, pP) in XYZ order
    quality : str
        Trade-off level: 'fast', 'balanced', 'high'
        - 'fast': Fewer iterations, lower border quality, larger tiles
        - 'balanced': Good quality with reasonable speed (default)
        - 'high': Maximum quality, smaller tiles, more iterations
    max_memory_gb : float, optional
        Maximum memory to use (GB). Defaults to 50% of available RAM.

    Returns
    -------
    DeconvolutionConfig
        Optimized configuration
    """
    M, N, P = im.shape
    pM, pN, pP = psf.shape

    # Determine available memory
    if max_memory_gb is None:
        if HAS_PSUTIL:
            available_gb = psutil.virtual_memory().available / (1024**3)
            max_memory_gb = available_gb * 0.5  # Use 50% of available
        else:
            max_memory_gb = 4.0  # Conservative default if psutil not available

    # Quality presets
    quality_params = {
        "fast": {
            "n_iter": 10,
            "border_quality": 1,
            "tile_memory_factor": 1.5,  # Larger tiles
        },
        "balanced": {
            "n_iter": 20,
            "border_quality": 2,
            "tile_memory_factor": 1.0,
        },
        "high": {
            "n_iter": 30,
            "border_quality": 2,
            "tile_memory_factor": 0.7,  # Smaller tiles for safety
        },
    }

    params = quality_params.get(quality, quality_params["balanced"])

    # Auto-determine tile overlap based on PSF size
    # Overlap should be at least PSF_radius + some margin for Bertero weights
    psf_max_radius = max(pM, pN) // 2
    tile_overlap = min(50, max(20, psf_max_radius + 10))

    # Determine if tiling is needed
    estimated_mem = estimate_memory_usage(
        im.shape, psf.shape, params["border_quality"], params["n_iter"]
    )

    needs_tiling = estimated_mem > max_memory_gb
    tile_max_size = None

    if needs_tiling:
        # Calculate optimal tile size
        # Target: tile memory ≈ max_memory_gb / tile_memory_factor
        target_tile_mem = max_memory_gb / params["tile_memory_factor"]

        # Binary search for tile size that fits in memory
        # Start with a guess based on image size
        tile_fraction = np.sqrt(target_tile_mem / estimated_mem)
        initial_tile = int(max(M, N) * tile_fraction)

        # Round to nearest 128 for FFT efficiency
        tile_max_size = max(256, (initial_tile // 128) * 128)

        print(f"Image size: {M}×{N}×{P}")
        print(f"Estimated memory for full image: {estimated_mem:.2f} GB")
        print(f"Available memory: {max_memory_gb:.2f} GB")
        print(f"Using tiling with tile_max_size={tile_max_size}, overlap={tile_overlap}")
    else:
        print(f"Image fits in memory ({estimated_mem:.2f} GB < {max_memory_gb:.2f} GB)")
        print("Processing without tiling")

    # Auto-determine offset based on image background
    # Use median of lowest 1% of pixels as estimate
    im_flat = im.ravel()
    bg_estimate = float(np.percentile(im_flat, 1))
    offset = max(0.0, bg_estimate * 0.5)  # Use half of background estimate

    # Create config
    cfg = DeconvolutionConfig(
        n_iter=params["n_iter"],
        border_quality=params["border_quality"],
        tile_max_size=tile_max_size,
        tile_overlap=tile_overlap,
        positivity=True,
        metric="idiv",
        start_condition="flat",
        xycropfactor=0.001,
        use_weights=True,  # Always use Bertero weights for accuracy
        offset=offset,
        pad_fast_fft=True,  # Always enable for speed
        alphamax=1.0,
        stop_rel=0.001,  # Auto-stop when converged
    )

    return cfg


def recommend_backend(available_backends: Optional[list] = None) -> str:
    """
    Recommend the fastest available backend.

    Parameters
    ----------
    available_backends : list, optional
        List of available backends. If None, will check automatically.

    Returns
    -------
    str
        Recommended backend name
    """
    if available_backends is None:
        available_backends = []

        # Check JAX
        try:
            import jax
            available_backends.append("jax")
            # Check if GPU is available
            try:
                if jax.devices("gpu"):
                    print("JAX GPU detected - recommended for best performance")
                else:
                    print("JAX CPU available")
            except:
                print("JAX CPU available")
        except ImportError:
            pass

        # Check Numba
        try:
            import numba
            available_backends.append("numba")
            print("Numba available")
        except ImportError:
            pass

        # Check FFTW
        try:
            import pyfftw
            available_backends.append("fftw")
            print("PyFFTW available")
        except ImportError:
            pass

        # NumPy is always available
        available_backends.append("numpy")

    # Return in order of preference
    preference_order = ["jax", "numba", "fftw", "numpy"]
    for backend in preference_order:
        if backend in available_backends:
            return backend

    return "numpy"


def auto_deconvolve(
    im: np.ndarray,
    psf: np.ndarray,
    method: str = "shb",
    quality: str = "balanced",
    backend: Optional[str] = None,
    max_memory_gb: Optional[float] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Fully automatic deconvolution with smart parameter selection.

    Parameters
    ----------
    im : np.ndarray
        Input image (M, N, P) in XYZ order
    psf : np.ndarray
        Point spread function
    method : str
        'rl' or 'shb' (default: 'shb')
    quality : str
        'fast', 'balanced', or 'high'
    backend : str, optional
        Backend to use. If None, automatically selects fastest available.
    max_memory_gb : float, optional
        Maximum memory budget in GB
    verbose : bool
        Print parameter selection details

    Returns
    -------
    np.ndarray
        Deconvolved image
    """
    if verbose:
        print("=== Auto-Deconvolution Configuration ===")

    # Get optimal config
    cfg = auto_config(im, psf, quality=quality, max_memory_gb=max_memory_gb)

    # Select backend
    if backend is None:
        backend = recommend_backend()

    if verbose:
        print(f"\nSelected backend: {backend}")
        print(f"Method: {method}")
        print(f"Iterations: {cfg.n_iter}")
        print(f"Border quality: {cfg.border_quality}")
        print(f"Auto-stop relative threshold: {cfg.stop_rel}")
        print("=" * 40)

    # Import and run
    if cfg.tile_max_size is not None:
        from .dw_numpy import deconvolve_tiled
        return deconvolve_tiled(im, psf, method=method, cfg=cfg)
    elif backend == "numpy":
        from .dw_numpy import deconvolve
        return deconvolve(im, psf, method=method, cfg=cfg)
    else:
        from .dw_fast import deconvolve_fast
        return deconvolve_fast(im, psf, method=method, backend=backend, cfg=cfg)


__all__ = ["auto_config", "recommend_backend", "auto_deconvolve", "estimate_memory_usage"]

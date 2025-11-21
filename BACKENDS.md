# dwpy Backend Architecture

## Overview

dwpy uses a **unified, backend-agnostic implementation** where all backends (NumPy, JAX, CuPy, Numba, FFTW) share the same deconvolution algorithm from the reference NumPy implementation. This ensures:

1. **Consistency**: All backends produce identical results (within floating-point precision)
2. **Maintainability**: Bug fixes and improvements only need to be made once
3. **Flexibility**: Easy to add new backends (e.g., PyTorch, TensorFlow)

## Backend Files

### Core Implementation

- **`dw_numpy.py`**: Reference implementation using pure NumPy
  - Contains `deconvolve()` and `deconvolve_tiled()` functions
  - Defines `DeconvolutionConfig` dataclass
  - All other backends mirror this implementation

- **`dw_fast.py`**: Backend dispatcher and unified implementation
  - `deconvolve_fast(backend='...')` - main entry point for all backends
  - `_deconvolve_backend()` - backend-agnostic implementation
  - Supports: `'numpy'`, `'jax'`, `'cupy'`, `'numba'`, `'fftw'`
  - Automatically handles array conversion and FFT selection

### Backend-Specific Helpers

- **`dw_numba.py`**: Numba JIT-compiled kernels for CPU acceleration
  - Provides `numba_kernels()` with optimized IDIV calculation

- **`dw_fftw.py`**: FFTW plan caching and threading
  - `fftw_plan()` creates cached FFT plans with threading support

- **`dw_dask.py`**: Dask wrapper for distributed/out-of-core processing
  - `dask_deconvolve()` handles chunking and overlap
  - Works with any backend

### Auto-Configuration

- **`dw_auto.py`**: Automatic parameter selection
  - `auto_deconvolve()` - fully automatic deconvolution
  - `auto_config()` - parameter optimization
  - `recommend_backend()` - detects and recommends fastest backend
  - `estimate_memory_usage()` - memory planning for tiling

### Legacy Files

- **`dw_jax.py`**: ⚠️ **LEGACY** - Old standalone JAX implementation
  - Predates the unified dw_fast approach
  - **Not recommended for new code**
  - Use `deconvolve_fast(backend='jax')` instead
  - Contains `run_dw()`, `run_dw_tiled()` with different API

- **`dexp_deconv.py`**: DEXP library integration (external dependency)
  - Only for users who already use DEXP
  - Has many external dependencies (dexp, xarray, etc.)

## Backend Comparison

| Backend | Implementation | Speed | GPU | Dependencies | Notes |
|---------|---------------|-------|-----|--------------|-------|
| **numpy** | Pure NumPy | 1x (baseline) | No | numpy, scipy | Reference, always works |
| **jax** | JAX with XLA | 3-4x | Yes | jax, jaxlib | Best for GPU/TPU, CPU fallback |
| **cupy** | CuPy GPU | 3-4x | Yes | cupy | NVIDIA GPUs only |
| **numba** | Numba JIT | 1-2x | No | numba | Good single-core CPU |
| **fftw** | PyFFTW | 1-1.5x | No | pyfftw | Multi-threaded FFTs |
| **dask** | Dask + any backend | Varies | Optional | dask | Distributed/out-of-core |

## How Backends Work

All backends in `dw_fast.py` follow this pattern:

```python
def deconvolve_fast(im, psf, method='shb', backend='numpy', cfg=None):
    # 1. Import backend-specific array library (xp)
    if backend == 'jax':
        import jax.numpy as jnp
        xp = jnp
        to_host = lambda x: np.array(device_get(x))

    # 2. Call unified backend implementation
    return _deconvolve_backend(im, psf, method, cfg, xp, to_host)
```

### Unified Backend Implementation

`_deconvolve_backend()` uses the array library `xp` (could be `numpy`, `jnp`, `cupy`, etc.):

```python
def _deconvolve_backend(im, psf, method, cfg, xp, to_host, fftr=None, kernels=None):
    # Convert input to backend arrays
    im = xp.asarray(im, dtype=xp.float32)
    psf = xp.asarray(psf, dtype=xp.float32)

    # Use backend's FFT (or custom fftr)
    if fftr is None:
        fftr = (xp.fft.rfftn, xp.fft.irfftn)

    # Run algorithm (mirrors dw_numpy.deconvolve exactly)
    # ... deconvolution iterations ...

    # Convert back to NumPy for output
    return to_host(result)
```

This ensures all backends use **identical math** with only the compute backend changed.

## Usage Examples

### Basic Usage
```python
import dwpy

# NumPy (CPU)
result = dwpy.deconvolve_fast(im, psf, backend='numpy')

# JAX (GPU/TPU if available, else CPU)
result = dwpy.deconvolve_fast(im, psf, backend='jax')

# CuPy (NVIDIA GPU)
result = dwpy.deconvolve_fast(im, psf, backend='cupy')

# Numba (JIT-compiled CPU)
result = dwpy.deconvolve_fast(im, psf, backend='numba')

# FFTW (multi-threaded FFTs)
result = dwpy.deconvolve_fast(im, psf, backend='fftw')
```

### Automatic Backend Selection
```python
# Auto-select fastest available backend
result = dwpy.auto_deconvolve(im, psf, quality='balanced')

# Or manually check
backend = dwpy.recommend_backend()
print(f"Using backend: {backend}")
result = dwpy.deconvolve_fast(im, psf, backend=backend)
```

### Dask for Large Images
```python
# Process large images with any backend
result = dwpy.dask_deconvolve(
    im, psf,
    backend='jax',      # Use JAX on each chunk
    chunk_xy=512,       # Chunk size
    overlap=50,         # Overlap for seamless stitching
)
```

## Testing Backend Consistency

All backends produce identical results (within floating-point precision):

```python
import numpy as np

result_numpy = dwpy.deconvolve_fast(im, psf, backend='numpy')
result_jax = dwpy.deconvolve_fast(im, psf, backend='jax')

max_diff = np.abs(result_numpy - result_jax).max()
print(f"Max difference: {max_diff:.2e}")  # Typically < 1e-5
```

## Adding New Backends

To add a new backend (e.g., PyTorch):

1. Import the array library in `deconvolve_fast()`
2. Define `to_host()` function to convert to NumPy
3. Optionally provide custom FFT functions
4. Call `_deconvolve_backend()` with the array library

Example:
```python
if backend == "torch":
    import torch
    xp = torch
    to_host = lambda x: x.cpu().numpy()
    return _deconvolve_backend(im, psf, method, cfg, xp, to_host)
```

The unified implementation handles the rest!

## Performance Tips

1. **GPU Workloads**: Use `jax` or `cupy` for best GPU performance
2. **CPU Only**: Use `numba` for single-core, `fftw` for multi-threaded
3. **Large Images**: Use `dask_deconvolve()` with tiling
4. **Memory Limited**: Set `cfg.tile_max_size` for automatic tiling
5. **Quality vs Speed**: Adjust `cfg.n_iter` (10=fast, 30=high quality)

## Troubleshooting

### Backend Not Available
```python
try:
    result = dwpy.deconvolve_fast(im, psf, backend='jax')
except ImportError as e:
    print(f"JAX not installed: {e}")
    # Fall back to numpy
    result = dwpy.deconvolve_fast(im, psf, backend='numpy')
```

### Out of Memory
```python
# Enable tiling
cfg = dwpy.DeconvolutionConfig(
    tile_max_size=512,
    tile_overlap=50,
)
result = dwpy.deconvolve_tiled(im, psf, cfg=cfg)

# Or use Dask
result = dwpy.dask_deconvolve(im, psf, chunk_xy=512)
```

### Inconsistent Results
- Check dtype (should be float32)
- Verify PSF normalization: `psf = psf / psf.sum()`
- Ensure identical `DeconvolutionConfig` parameters
- Small differences (< 1e-5) are normal due to floating-point precision

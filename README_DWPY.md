# dwpy - DeconWolf Python Package

Fast, flexible deconvolution backends for microscopy image processing.

## Overview

`dwpy` provides multiple high-performance backends for Richardson-Lucy (RL) and Scaled Heavy Ball (SHB) deconvolution:

- **NumPy**: Reference implementation, works everywhere
- **JAX**: GPU/TPU-accelerated, 3-4x faster than NumPy on GPU
- **CuPy**: NVIDIA GPU-accelerated using CuPy, similar performance to JAX
- **Numba**: JIT-compiled Python, good CPU performance
- **FFTW**: Optimized FFT library integration
- **Dask**: Distributed/out-of-core processing for large images

## Installation

### Basic Installation
```bash
pip install -e .
```

### With Optional Backends

Install with specific backends:
```bash
# JAX for GPU/TPU support
pip install -e ".[jax]"

# CuPy for NVIDIA GPU support
pip install -e ".[cupy]"

# Numba for fast CPU
pip install -e ".[numba]"

# FFTW for optimized FFTs
pip install -e ".[fftw]"

# Dask for distributed processing
pip install -e ".[dask]"
```

Install with all backends:
```bash
pip install -e ".[all]"
```

Install for development:
```bash
pip install -e ".[dev]"
```

## Quick Start

### Simple Deconvolution
```python
import numpy as np
import dwpy

# Load your image and PSF
im = ...  # 3D array (Z, Y, X) or (X, Y, Z)
psf = ... # 3D PSF array

# Deconvolve with default settings
result = dwpy.deconvolve(im, psf, method='shb')
```

### Automatic Configuration
```python
from dwpy.dw_auto import auto_deconvolve

# Automatically selects best backend and parameters
result = auto_deconvolve(
    im,
    psf,
    quality='balanced',  # 'fast', 'balanced', or 'high'
    backend=None,        # Auto-selects fastest available
    max_memory_gb=8.0    # Memory budget
)
```

### Fast Backend Selection
```python
from dwpy.dw_fast import deconvolve_fast

# Use specific backend
result = deconvolve_fast(
    im,
    psf,
    method='shb',
    backend='jax'  # 'numpy', 'jax', 'cupy', 'numba', or 'fftw'
)
```

### Advanced Configuration
```python
from dwpy import DeconvolutionConfig, deconvolve

cfg = DeconvolutionConfig(
    n_iter=30,              # Number of iterations
    border_quality=2,       # Border handling (0, 1, 2)
    positivity=True,        # Enforce positive values
    metric='idiv',          # Cost function ('idiv' or 'ssd')
    use_weights=True,       # Use Bertero weights
    stop_rel=0.001,         # Auto-stop threshold
)

result = deconvolve(im, psf, method='shb', cfg=cfg)
```

### Tiled Processing (Large Images)
```python
from dwpy import deconvolve_tiled, DeconvolutionConfig

cfg = DeconvolutionConfig(
    n_iter=20,
    tile_max_size=512,     # Maximum tile dimension
    tile_overlap=50,       # Overlap between tiles
)

result = deconvolve_tiled(im, psf, method='shb', cfg=cfg)
```

## Methods

- **'rl'**: Richardson-Lucy deconvolution
- **'shb'**: Scaled Heavy Ball method (recommended, faster convergence)

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_iter` | int | 20 | Number of iterations |
| `border_quality` | int | 2 | Border quality (0=none, 1=half, 2=full) |
| `positivity` | bool | True | Enforce positive values |
| `metric` | str | 'idiv' | Cost function ('idiv' or 'ssd') |
| `use_weights` | bool | True | Use Bertero edge weights |
| `tile_max_size` | int\|None | None | Enable tiling for large images |
| `tile_overlap` | int | 50 | Overlap between tiles |
| `stop_rel` | float | 0.001 | Relative change threshold for auto-stop |
| `offset` | float | 0.0 | Background offset subtraction |
| `pad_fast_fft` | bool | True | Pad to fast FFT sizes |

## Backends Comparison

| Backend | Speed | GPU | Notes |
|---------|-------|-----|-------|
| **NumPy** | 1x | No | Reference, always available |
| **JAX** | 3-4x | Yes | Best performance, supports GPU/TPU/CPU |
| **CuPy** | 3-4x | Yes | NVIDIA GPU only, similar to JAX |
| **Numba** | 1-2x | No | JIT-compiled CPU, good single-core performance |
| **FFTW** | 1-1.5x | No | Optimized FFTs with threading |
| **Dask** | Varies | Optional | For distributed/out-of-core processing |

## PSF Generation

dwpy provides two PSF models: **Born-Wolf** (simpler, faster) and **Gibson-Lanni** (more accurate with RI mismatch).

### Born-Wolf PSF (Recommended for most cases)

```python
import dwpy

# Generate Born-Wolf PSF
psf = dwpy.generate_psf_bw(
    dxy=0.065,             # Lateral pixel size (μm)
    dz=0.2,                # Axial pixel size (μm)
    xy_size=64,            # Lateral size (pixels, should be odd)
    z_size=64,             # Axial size (pixels, should be odd)
    NA=1.4,                # Numerical aperture
    ni=1.518,              # Immersion medium RI
    wvl=0.520,             # Emission wavelength (μm)
)
```

### Gibson-Lanni PSF (For RI mismatch scenarios)

Use when imaging aqueous specimens (cells, water) with oil immersion objectives:

```python
import dwpy

# Generate Gibson-Lanni PSF
# 60x/1.4NA oil immersion imaging into aqueous specimen
psf = dwpy.generate_psf_gl(
    dxy=0.065,             # Lateral pixel size (μm)
    dz=0.2,                # Axial pixel size (μm)
    xy_size=64,            # Lateral size (pixels)
    z_size=64,             # Axial size (pixels)
    NA=1.4,                # Numerical aperture
    ni=1.515,              # Immersion RI (oil)
    ns=1.33,               # Specimen RI (water/cells)
    wvl=0.520,             # Emission wavelength (μm)
    M=60,                  # Magnification
    ti0=150,               # Working distance (μm)
    # Optional: coverslip parameters
    tg=170,                # Coverslip thickness (μm)
    ng=1.515,              # Coverslip RI
)
```

**When to use which:**
- **Born-Wolf**: Simpler, faster, good for matched RI (e.g., oil immersion into oil-matched sample)
- **Gibson-Lanni**: More accurate when specimen RI ≠ immersion RI (e.g., oil into water/cells), accounts for coverslip and working distance effects

## Examples

See the `demo/` directory for complete examples:

- `demo/benchmark_all.py` - Performance comparison
- `demo/run_all_backends.py` - Run all backends
- `demo/benchmark_synthetic.py` - Quality assessment with ground truth

## API Reference

### Main Functions

- `deconvolve(im, psf, method='shb', cfg=None)` - Basic deconvolution
- `deconvolve_tiled(im, psf, method='shb', cfg=None)` - Tiled processing
- `deconvolve_fast(im, psf, method='shb', backend='jax', cfg=None)` - Backend selection
- `auto_deconvolve(im, psf, quality='balanced', ...)` - Automatic configuration

### Configuration

- `DeconvolutionConfig(...)` - Configuration dataclass
- `auto_config(im, psf, quality='balanced', ...)` - Auto parameter selection
- `recommend_backend()` - Detect best available backend
- `estimate_memory_usage(...)` - Estimate memory requirements

### PSF Tools

- `generate_psf_bw(...)` - Generate Born-Wolf PSF
- `generate_psf_gl(...)` - Generate Gibson-Lanni PSF

## Performance Tips

1. **Use JAX with GPU** for best performance (3-4x speedup)
2. **Enable auto-stop** with `stop_rel=0.001` to avoid unnecessary iterations
3. **Use tiling** for images that don't fit in memory
4. **Adjust border_quality** based on needs (2=best quality, 0=fastest)
5. **Use SHB method** over RL for faster convergence

## Citation

If you use this software, please cite:

```
DeconWolf: Fast and robust microscopy image deconvolution
https://github.com/elgw/deconwolf
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! See the main repository for guidelines.

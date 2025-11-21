"""
Dask-friendly wrapper for deconvolution.

Sets sensible defaults for chunking/overlap and reuses per-worker caches
for FFT plans/JITs.
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import dask.array as da
import numpy as np

from .dw_fast import deconvolve_fast, Backend, Method
from .dw_numpy import DeconvolutionConfig


def dask_deconvolve(
    im: np.ndarray,
    psf: np.ndarray,
    *,
    backend: Backend = "numpy",
    method: Method = "shb",
    chunk_xy: int = 256,
    overlap: Optional[int] = None,
    cfg: Optional[DeconvolutionConfig] = None,
) -> np.ndarray:
    """
    Run deconvolution over a Dask array with overlap handling.

    Args:
        im: 3D numpy array; will be chunked in XY.
        psf: PSF array.
        backend: 'numpy'/'cupy'/'jax'/'numba'/'fftw'.
        chunk_xy: chunk size in X and Y (Z unchunked).
        overlap: overlap in X/Y; defaults to half PSF size.
        cfg: DeconvolutionConfig (tile settings are ignored here).
    """
    cfg = cfg or DeconvolutionConfig()
    M, N, P = im.shape
    if overlap is None:
        overlap = max(psf.shape[0] // 2, psf.shape[1] // 2)
    chunks = (chunk_xy, chunk_xy, P)

    darr = da.from_array(im, chunks=chunks)
    func = partial(_dask_block, psf=psf, backend=backend, method=method, cfg=cfg)
    out = darr.map_overlap(func, depth=(overlap, overlap, 0), boundary="reflect", dtype=im.dtype)
    return out.compute()


def _dask_block(block, psf, backend, method, cfg):
    # Ensure contiguous float32
    block = np.asarray(block, dtype=np.float32)
    return deconvolve_fast(block, psf, method=method, backend=backend, cfg=cfg)


__all__ = ["dask_deconvolve"]

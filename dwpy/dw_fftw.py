"""
pyFFTW helpers: plan-cached FFT functions that mirror numpy.fft signatures.

Usage:
    fftn, ifftn = fftw_plan((M,N,P))
    F = fftn(x)
    x = ifftn(F)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Tuple

import numpy as np

try:
    import pyfftw  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pyfftw = None


def fftw_plan(shape: Tuple[int, int, int], threads: int = 1) -> Tuple[Callable, Callable]:
    if pyfftw is None:
        raise ImportError("pyfftw is required for fftw_plan")

    # cache aligned arrays and plans keyed by shape+threads
    key = (shape, threads)
    fftn = _fftn_cached(key)
    ifftn = _ifftn_cached(key)
    return fftn, ifftn


@lru_cache(maxsize=32)
def _fftn_cached(key):
    shape, threads = key
    norm = float(np.prod(shape))
    a = pyfftw.empty_aligned(shape, dtype="float32")
    out = pyfftw.empty_aligned((shape[0], shape[1], shape[2] // 2 + 1), dtype="complex64")
    plan = pyfftw.builders.rfftn(a, s=shape, threads=threads)

    def fftn(x, s=None):
        if s is not None and tuple(s) != tuple(shape):
            raise ValueError(f"fftn shape {s} does not match planned shape {shape}")
        out[:] = plan(x, normalise_idft=False)
        return (out * norm).copy()

    return fftn


@lru_cache(maxsize=32)
def _ifftn_cached(key):
    shape, threads = key
    norm = float(np.prod(shape))
    a = pyfftw.empty_aligned((shape[0], shape[1], shape[2] // 2 + 1), dtype="complex64")
    out = pyfftw.empty_aligned(shape, dtype="float32")
    plan = pyfftw.builders.irfftn(a, s=shape, threads=threads)

    def ifftn(x, s=None):
        if s is not None and tuple(s) != tuple(shape):
            raise ValueError(f"ifftn shape {s} does not match planned shape {shape}")
        out[:] = plan(x, normalise_idft=False)
        return (out / norm).copy()

    return ifftn


__all__ = ["fftw_plan"]

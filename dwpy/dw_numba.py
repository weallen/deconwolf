"""
Numba-jitted elementwise kernels used by the 'numba' backend in dw_fast.
"""

from __future__ import annotations

import numpy as np


def numba_kernels():
    """Return lazily-imported Numba kernels; raises ImportError if Numba is missing."""
    try:
        from numba import njit, prange
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Numba backend requested but numba is not installed") from exc

    @njit(parallel=True, fastmath=True)
    def idiv_numba(obs: np.ndarray, est: np.ndarray, mask: np.ndarray, denom: int) -> float:
        acc = 0.0
        obs_flat = obs.ravel()
        est_flat = est.ravel()
        mask_flat = mask.ravel()
        for i in prange(obs_flat.size):
            if mask_flat[i]:
                acc += est_flat[i] * np.log(est_flat[i] / obs_flat[i]) - (est_flat[i] - obs_flat[i])
        return acc / denom

    return {"idiv": idiv_numba}


__all__ = ["numba_kernels"]

"""
Parity checks between NumPy and JAX deconvolution on a small synthetic volume.

These tests ensure the reference NumPy implementation stays aligned with the
JAX backend for both RL and SHB update rules. They use a tiny Gaussian PSF and
synthetic data to keep runtime low.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from dwpy import dw_fast, dw_jax, dw_numpy  # noqa: E402


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized correlation of two arrays."""
    a = a.ravel()
    b = b.ravel()
    return float(np.corrcoef(a, b)[0, 1])


def _make_data(shape=(16, 16, 8)) -> tuple[np.ndarray, np.ndarray]:
    """Create a small synthetic image and Gaussian PSF."""
    rng = np.random.default_rng(0)
    im = rng.poisson(5.0, size=shape).astype(np.float32)
    # Add a bright blob near the center to give structure.
    cx, cy, cz = (np.array(shape) // 2).astype(int)
    im[cx, cy, cz] += 50.0
    psf = dw_numpy.gaussian_kernel_3d(1.0, 1.0, 1.0)
    return im, psf


def test_rl_parity_numpy_vs_jax():
    im, psf = _make_data()
    cfg = dw_numpy.DeconvolutionConfig(
        n_iter=6,
        border_quality=2,
        positivity=True,
        use_weights=False,
    )
    out_np = dw_numpy.deconvolve(im, psf, method="rl", cfg=cfg)

    out_j = dw_jax.run_dw(
        jnp.array(im),
        jnp.array(psf),
        n_iter=6,
        border_quality=2,
        positivity=True,
        method="rl",
        verbose=False,
    )
    out_j = np.array(out_j)

    corr = _corr(out_np, out_j)
    assert corr > 0.99, f"RL parity too low: corr={corr:.4f}"


def test_shb_parity_numpy_vs_jax():
    im, psf = _make_data()
    cfg = dw_numpy.DeconvolutionConfig(
        n_iter=8,
        border_quality=2,
        positivity=True,
        use_weights=False,
    )
    out_np = dw_numpy.deconvolve(im, psf, method="shb", cfg=cfg)

    out_j = dw_jax.run_dw(
        jnp.array(im),
        jnp.array(psf),
        n_iter=8,
        border_quality=2,
        positivity=True,
        method="shb_jit",
        verbose=False,
    )
    out_j = np.array(out_j)

    corr = _corr(out_np, out_j)
    assert corr > 0.98, f"SHB parity too low: corr={corr:.4f}"


def test_shb_parity_fast_jax_vs_dw_jax():
    im, psf = _make_data()
    cfg = dw_numpy.DeconvolutionConfig(
        n_iter=8,
        border_quality=2,
        positivity=True,
        use_weights=False,
    )
    out_fast = dw_fast.deconvolve_fast(im, psf, method="shb", backend="jax", cfg=cfg)

    out_j = dw_jax.run_dw(
        jnp.array(im),
        jnp.array(psf),
        n_iter=8,
        border_quality=2,
        positivity=True,
        method="shb_jit",
        verbose=False,
    )
    out_j = np.array(out_j)
    corr = _corr(out_fast, out_j)
    assert corr > 0.98, f"Fast JAX parity too low: corr={corr:.4f}"

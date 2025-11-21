import numpy as np
import pytest

from dwpy import (
    DeconvolutionConfig,
    deconvolve,
    deconvolve_fast,
    deconvolve_tiled,
    dask_deconvolve,
)


def _make_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    im = rng.random((12, 12, 5), dtype=np.float32) * 10 + 1.0
    psf = np.zeros((7, 7, 5), dtype=np.float32)
    psf[3, 3, 2] = 1.0
    # small Gaussian blur of the PSF for realism
    psf = psf / psf.sum()
    return im, psf


def _cfg(**kwargs):
    base = dict(
        n_iter=2,
        bg=None,
        pad_fast_fft=False,
        start_condition="flat",
        border_quality=2,
        metric="mse",
    )
    base.update(kwargs)
    return DeconvolutionConfig(**base)


def _close(a, b, rtol=1e-3, atol=1e-4):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


def test_numpy_baseline_and_padded():
    im, psf = _make_data()
    cfg = _cfg()
    out = deconvolve(im, psf, cfg=cfg)
    cfg_pad = _cfg(pad_fast_fft=True)
    out_pad = deconvolve(im, psf, cfg=cfg_pad)
    _close(out, out_pad, rtol=5e-3, atol=1e-3)


def test_tiled_matches_full():
    im, psf = _make_data()
    cfg = _cfg(n_iter=2, tile_max_size=8, tile_overlap=3)
    out_tiled = deconvolve_tiled(im, psf, cfg=cfg)
    out_full = deconvolve(im, psf, cfg=_cfg(n_iter=2))
    _close(out_full, out_tiled, rtol=5e-3, atol=1e-3)


def test_numba_backend_matches_numpy():
    pytest.importorskip("numba")
    im, psf = _make_data()
    cfg = _cfg(n_iter=2)
    out_np = deconvolve_fast(im, psf, backend="numpy", cfg=cfg)
    out_nb = deconvolve_fast(im, psf, backend="numba", cfg=cfg)
    _close(out_np, out_nb, rtol=5e-3, atol=1e-3)


def test_fftw_backend_matches_numpy():
    pyfftw = pytest.importorskip("pyfftw")
    # silence unused warning
    _ = pyfftw
    im, psf = _make_data()
    cfg = _cfg(n_iter=2, pad_fast_fft=True, fftw_threads=1)
    out_np = deconvolve_fast(im, psf, backend="numpy", cfg=_cfg(n_iter=2))
    out_fftw = deconvolve_fast(im, psf, backend="fftw", cfg=cfg)
    _close(out_np, out_fftw, rtol=5e-3, atol=1e-3)


def test_jax_backend_matches_numpy():
    jax = pytest.importorskip("jax")
    _ = jax
    im, psf = _make_data()
    cfg = _cfg(n_iter=1, pad_fast_fft=False, jax_platform="cpu")
    out_np = deconvolve_fast(im, psf, backend="numpy", cfg=cfg)
    out_jax = deconvolve_fast(im, psf, backend="jax", cfg=cfg)
    _close(out_np, out_jax, rtol=5e-3, atol=1e-3)


def test_dask_wrapper_matches_numpy():
    pytest.importorskip("dask")
    im, psf = _make_data()
    cfg = _cfg(n_iter=1)
    out_np = deconvolve_fast(im, psf, backend="numpy", cfg=cfg)
    out_dask = dask_deconvolve(im, psf, backend="numpy", method="shb", chunk_xy=8, overlap=3, cfg=cfg)
    _close(out_np, out_dask, rtol=5e-3, atol=1e-3)

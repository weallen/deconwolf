"""
Backend-selectable deconvolution: NumPy (reference), CuPy, and JAX.

The CuPy/JAX paths mirror the reference logic in ``dw_numpy.deconvolve``
so they stay behaviorally aligned while providing faster FFTs and
elementwise math when GPUs/accelerators are available.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional, Tuple

import numpy as np

from .dw_numpy import DeconvolutionConfig, deconvolve as deconvolve_numpy, work_shape, next_fast_shape
from .dw_fftw import fftw_plan

Backend = Literal["numpy", "cupy", "jax", "numba", "fftw"]
Method = Literal["rl", "shb"]


def deconvolve_fast(
    im: np.ndarray,
    psf: np.ndarray,
    method: Method = "shb",
    backend: Backend = "numpy",
    cfg: Optional[DeconvolutionConfig] = None,
) -> np.ndarray:
    """
    Deconvolve with a selectable backend.

    backend:
      - 'numpy': CPU reference (dw_numpy.deconvolve)
      - 'cupy': GPU using CuPy FFTs/elementwise ops
      - 'jax' : Accelerated JAX (CPU/GPU/TPU depending on default device)
      - 'numba': CPU with NumPy FFTs + Numba-jitted elementwise kernels
      - 'fftw': CPU with pyFFTW FFTs and plan cache
    """
    if backend == "numpy":
        return deconvolve_numpy(im, psf, method=method, cfg=cfg)

    if backend == "cupy":
        try:
            import cupy as cp
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("CuPy backend requested but cupy is not installed") from exc
        return _deconvolve_backend(im, psf, method, cfg, cp, lambda x: cp.asnumpy(x))

    if backend == "jax":
        try:
            import jax.numpy as jnp
            import jax
            from jax import device_get  # noqa: WPS433 - re-exported helper
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("JAX backend requested but jax is not installed") from exc
        if cfg and cfg.jax_platform:
            jax.config.update("jax_platform_name", cfg.jax_platform)
        # thread hints can be set via env; we leave to user or cfg.jax_num_threads if desired
        return _deconvolve_backend(im, psf, method, cfg, jnp, lambda x: np.array(device_get(x)))

    if backend == "numba":
        try:
            import numpy as xp
            from .dw_numba import numba_kernels
            kernels = numba_kernels()
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Numba backend requested but numba is not installed") from exc
        return _deconvolve_backend(im, psf, method, cfg, xp, lambda x: x, fftr=(xp.fft.rfftn, xp.fft.irfftn), kernels=kernels)

    if backend == "fftw":
        try:
            import pyfftw  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("FFTW backend requested but pyfftw is not installed") from exc
        xp = np
        plan_factory = lambda shape: fftw_plan(shape, threads=cfg.fftw_threads if cfg else 1)
        return _deconvolve_backend(im, psf, method, cfg, xp, lambda x: x, fftr=plan_factory, kernels=None)

    raise ValueError(f"Unknown backend '{backend}'")


# ---------------------- backend-generic helpers ---------------------- #
def _anscombe(x, xp):
    return 2.0 * xp.sqrt(x + xp.asarray(3.0 / 8.0, dtype=xp.float32))


def _inverse_anscombe(x, xp):
    return (x / 2.0) ** 2 - xp.asarray(3.0 / 8.0, dtype=xp.float32)


def _gaussian_kernel_3d(sx: float, sy: float, sz: float, xp):
    """Create a separable Gaussian kernel using NumPy then cast to xp."""
    def _axis_kernel(sigma: float) -> np.ndarray:
        radius = max(1, int(np.ceil(3.0 * sigma)))
        coords = np.arange(-radius, radius + 1, dtype=np.float32)
        kern = np.exp(-0.5 * (coords / sigma) ** 2)
        return kern / kern.sum()

    kx = _axis_kernel(sx)
    ky = _axis_kernel(sy)
    kz = _axis_kernel(sz)
    kernel = np.outer(kx, ky).reshape(kx.size, ky.size, 1) * kz.reshape(1, 1, kz.size)
    return xp.asarray(kernel, dtype=xp.float32)


def _fft_convolve_real(x, kernel, xp):
    """Same-size real convolution using xp FFTs."""
    target_shape = x.shape
    work = xp.zeros(target_shape, dtype=xp.float32)
    insert_shape = tuple(min(a, b) for a, b in zip(kernel.shape, target_shape))
    slices = tuple(slice(0, s) for s in insert_shape)
    if hasattr(work, "at"):
        work = work.at[slices].set(kernel[slices])
    else:
        work[slices] = kernel[slices]
    # center kernel
    shifts = tuple(-(dim // 2) for dim in work.shape)
    work = xp.roll(work, shift=shifts, axis=(0, 1, 2))
    kf = xp.fft.rfftn(work, s=target_shape)
    xf = xp.fft.rfftn(x, s=target_shape)
    return xp.fft.irfftn(xf * kf, s=target_shape).astype(xp.float32)


def _max_index(arr, to_host: Callable[[object], np.ndarray]) -> Tuple[int, int, int]:
    idx = int(np.nanargmax(to_host(arr)))
    return tuple(np.unravel_index(idx, arr.shape))


def _circshift(arr, shifts, xp):
    return xp.roll(arr, shift=shifts, axis=(0, 1, 2))


def _psf_autocrop_by_image(psf, im_shape, border_quality, xp, to_host):
    m, n, p = psf.shape
    M, N, P = im_shape
    if border_quality == 0:
        mopt, nopt, popt = M, N, P
    else:
        mopt, nopt, popt = (M - 1) * 2 + 1, (N - 1) * 2 + 1, (P - 1) * 2 + 1
    if p < popt:
        return psf

    m0 = n0 = p0 = 0
    m1, n1, p1 = m - 1, n - 1, p - 1
    if m > mopt:
        delta = m - mopt
        m0 = delta // 2
        m1 -= delta // 2
    if n > nopt:
        delta = n - nopt
        n0 = delta // 2
        n1 -= delta // 2
    if p > popt:
        delta = p - popt
        p0 = delta // 2
        p1 -= delta // 2
    return psf[m0 : m1 + 1, n0 : n1 + 1, p0 : p1 + 1]


def _psf_autocrop_xy(psf, xycropfactor, xp, to_host):
    m, n, p = psf.shape
    plane_sums = psf.sum(axis=(1, 2))
    maxsum = float(plane_sums.max())
    if maxsum <= 0:
        return psf
    first = 0
    # guard against infinite loop if factor too high
    while first < m and float(to_host(plane_sums[first])) < xycropfactor * maxsum:
        first += 1
    if first == 0 or first >= m // 2:
        return psf
    return psf[first : m - first, first : n - first, :]


def _psf_autocrop_center_z(psf, xp, to_host):
    # Keep parity with the C reference: Z centering is done via circshift
    # after padding, not by cropping planes here.
    return psf


def _psf_autocrop(psf, im_shape, border_quality, xycropfactor, xp, to_host):
    psf = _psf_autocrop_by_image(psf, im_shape, border_quality, xp, to_host)
    if border_quality > 0 and xycropfactor > 0:
        psf = _psf_autocrop_xy(psf, xycropfactor, xp, to_host)
    return psf


def _flatfield_and_offset(im, flatfield, offset, zcrop, xp):
    if zcrop > 0:
        if 2 * zcrop >= im.shape[2]:
            raise ValueError("zcrop too large for image depth")
        im = im[:, :, zcrop : im.shape[2] - zcrop]
    if flatfield is not None:
        im = im / xp.asarray(flatfield, dtype=im.dtype)
    if offset > 0:
        im = im + offset
    return im


def _pad_psf(psf, wshape, xp, to_host):
    """Pad PSF and circshift so its max lands at (0,0,0), mirroring C."""
    Z = xp.zeros(wshape, dtype=xp.float32)
    Z = _set_cube(Z, psf, psf.shape[0], psf.shape[1], psf.shape[2], xp)
    maxm, maxn, maxp = _max_index(Z, to_host)
    Z = _circshift(Z, (-maxm, -maxn, -maxp), xp)
    # sanity: ensure max is at origin
    if float(to_host(Z[0, 0, 0])) != float(to_host(Z).max()):
        raise RuntimeError("PSF centering failed to move max to origin")
    return Z


def _initial_guess_fft(im_shape, wshape, xp):
    one = xp.zeros(wshape, dtype=xp.float32)
    M, N, P = im_shape
    one = _set_cube(one, xp.asarray(1.0, dtype=xp.float32), M, N, P, xp)
    return xp.fft.rfftn(one, s=wshape)


def _compute_weights(cK, im_shape, wshape, xp):
    F_one = _initial_guess_fft(im_shape, wshape, xp)
    W = xp.fft.irfftn(xp.conj(cK) * F_one, s=wshape).astype(xp.float32)
    sigma = xp.asarray(0.01, dtype=xp.float32)
    mask = W > sigma
    out = xp.zeros_like(W, dtype=xp.float32)
    out = xp.where(mask, 1.0 / W, 0.0)
    return out


def _get_error(y, g, metric, xp, kernels=None):
    if metric == "mse":
        return _get_fmse(y, g, xp, kernels)
    return _get_idiv(y, g, xp, kernels)


def _get_fmse(y, g, xp, kernels=None):
    M, N, P = g.shape
    sub = y[:M, :N, :P]
    return float(sub.size and xp.mean((sub - g) ** 2))


def _get_idiv(y, g, xp, kernels=None):
    M, N, P = g.shape
    obs = y[:M, :N, :P]
    est = g
    mask = (obs > 0) & (est > 0)
    if not bool(mask.any()):
        return 0.0
    if kernels and "idiv" in kernels:
        return kernels["idiv"](obs, est, mask, M * N * P)
    val = est[mask] * xp.log(est[mask] / obs[mask]) - (est[mask] - obs[mask])
    return float(val.sum() / (M * N * P))


def _prefilter(im, psf, psigma, xp, kernels=None):
    if psigma <= 0:
        return im.astype(xp.float32), psf.astype(xp.float32)
    kernel = _gaussian_kernel_3d(psigma, psigma, psigma, xp)
    im_f = _inverse_anscombe(_fft_convolve_real(_anscombe(im.astype(xp.float32), xp), kernel, xp), xp)
    psf_f = _fft_convolve_real(psf.astype(xp.float32), kernel, xp)
    return im_f.astype(xp.float32), psf_f.astype(xp.float32)


def _iter_rl_step(im, fft_psf, f, W, bg, metric, xp, kernels=None, scratch=None):
    M, N, P = im.shape
    wshape = f.shape
    F = xp.fft.rfftn(f, s=wshape)
    y = xp.fft.irfftn(fft_psf * F, s=wshape).astype(xp.float32)
    error = _get_error(y, im, metric, xp, kernels)
    y_obs = y[:M, :N, :P]
    y_safe = xp.where(y_obs > 0, y_obs, bg)
    ratio = im / y_safe
    # fill outside FOV with small positive value to keep gradients alive
    y_full = xp.full_like(y, xp.asarray(1e-6, dtype=xp.float32))
    y_full = _set_cube(y_full, ratio, M, N, P, xp)
    F_sn = xp.fft.rfftn(y_full, s=wshape)
    x = xp.fft.irfftn(xp.conj(fft_psf) * F_sn, s=wshape).astype(xp.float32)
    if W is not None:
        x = x * f * W
    else:
        x = x * f
    return x, error


def _iter_shb_step(im, cK, pK, W, bg, metric, xp, kernels=None, scratch=None):
    M, N, P = im.shape
    wshape = pK.shape
    Pk = xp.fft.rfftn(pK, s=wshape)
    # Forward model: convolution with PSF (no conjugate), mirroring the C SHB reference
    y = xp.fft.irfftn(cK * Pk, s=wshape).astype(xp.float32)
    error = _get_error(y, im, metric, xp, kernels)
    mindiv = xp.asarray(1e-6, dtype=xp.float32)
    y_obs = y[:M, :N, :P]
    y_safe = xp.where(xp.abs(y_obs) < mindiv, xp.copysign(mindiv, y_obs), y_obs)
    ratio = im / y_safe
    # Outside observed region should be zero (C reference)
    y_full = xp.zeros_like(y, dtype=xp.float32)
    y_full = _set_cube(y_full, ratio, M, N, P, xp)
    Y = xp.fft.rfftn(y_full, s=wshape)
    x = xp.fft.irfftn(xp.conj(cK) * Y, s=wshape).astype(xp.float32)
    if W is not None:
        x = x * pK * W
    else:
        x = x * pK
    return x, error


def _set_cube(target, value, M, N, P, xp):
    # JAX needs functional updates; use slicing for numpy/cupy.
    return target.at[:M, :N, :P].set(value) if hasattr(target, "at") else _set_slice(target, value, M, N, P)


def _set_slice(target, value, M, N, P):
    target[:M, :N, :P] = value
    return target


def _deconvolve_backend(
    im,
    psf,
    method,
    cfg,
    xp,
    to_host: Callable[[object], np.ndarray],
    fftr=None,
    kernels=None,
) -> np.ndarray:
    cfg = cfg or DeconvolutionConfig()
    im_x = xp.asarray(im, dtype=xp.float32)
    psf_x = xp.asarray(psf, dtype=xp.float32)

    im_x = _flatfield_and_offset(im_x, cfg.flatfield, cfg.offset, cfg.zcrop, xp)

    # Calculate bg AFTER adding offset to match C reference behavior
    bg_val = cfg.bg
    if bg_val is None:
        bg_val = max(float(to_host(im_x).min()), 1e-2)
    bg = xp.asarray(bg_val, dtype=xp.float32)
    psf_x = psf_x / psf_x.sum()
    psf_x = _psf_autocrop(psf_x, im_x.shape, cfg.border_quality, cfg.xycropfactor, xp, to_host)
    psf_x = psf_x / psf_x.sum()
    im_x, psf_x = _prefilter(im_x, psf_x, cfg.psigma, xp, kernels)
    psf_x = psf_x / psf_x.sum()

    wshape = work_shape(im_x.shape, psf_x.shape, cfg.border_quality)
    if cfg.pad_fast_fft:
        wshape = next_fast_shape(wshape)
    if callable(fftr):
        fftn, ifftn = fftr(wshape)
    else:
        fftn, ifftn = fftr if fftr is not None else (xp.fft.rfftn, xp.fft.irfftn)
    cK = fftn(_pad_psf(psf_x, wshape, xp, to_host), s=wshape)
    W = None
    if cfg.border_quality > 0 and cfg.use_weights:
        W = _compute_weights(cK, im_x.shape, wshape, xp)

    M, N, P = im_x.shape
    wM, wN, wP = wshape
    sumg = float(to_host(im_x.sum()))
    if cfg.start_condition == "flat":
        x = xp.full(wshape, sumg / (wM * wN * wP), dtype=xp.float32)
    elif cfg.start_condition == "identity":
        x = xp.zeros(wshape, dtype=xp.float32)
        x = _set_cube(x, im_x, M, N, P, xp)
    elif cfg.start_condition == "lp":
        kernel = _gaussian_kernel_3d(cfg.start_lpsigma, cfg.start_lpsigma, cfg.start_lpsigma, xp)
        lp_im = _fft_convolve_real(im_x, kernel, xp)
        x = xp.zeros(wshape, dtype=xp.float32)
        x = _set_cube(x, lp_im, lp_im.shape[0], lp_im.shape[1], lp_im.shape[2], xp)
    else:
        raise ValueError(f"Unknown start_condition {cfg.start_condition}")

    xp_prev = x
    xp_prev2 = xp_prev
    prev_error = np.inf
    scratch = None

    for it in range(cfg.n_iter):
        if method == "shb":
            alpha = max(0.0, min(cfg.alphamax, (it - 1.0) / (it + 2.0)))
            p = x + alpha * (x - xp_prev)
            p = xp.where(p < bg, bg, p)
            if scratch is None or hasattr(xp, "device_put"):  # JAX case: avoid in-place reuse
                scratch = xp.zeros_like(x, dtype=xp.float32)
            xp_curr, err = _iter_shb_step(im_x, cK, p, W, bg, cfg.metric, xp, kernels, scratch)
            xp_prev = x  # Save current x as previous for next momentum calculation
            x = xp_curr  # Update x to new estimate
        else:
            current_f = xp_prev
            if cfg.biggs or cfg.eve:
                delta = xp_prev - xp_prev2
                beta = max(0.0, min(cfg.biggs_clip, it / (it + 3.0)))
                if cfg.eve:
                    beta = min(beta, 1.0 - np.exp(-(it + 1)))
                current_f = xp_prev + beta * delta
                current_f = xp.where(current_f < bg, bg, current_f)
            if scratch is None or hasattr(xp, "device_put"):  # JAX case
                scratch = xp.zeros_like(x, dtype=xp.float32)
            xp_curr, err = _iter_rl_step(im_x, cK, current_f, W, bg, cfg.metric, xp, kernels, scratch)
            xp_prev2 = xp_prev
            xp_prev = xp_curr
            x = xp_curr

        if cfg.stop_abs is not None and err < cfg.stop_abs:
            break
        if cfg.stop_rel is not None and np.isfinite(prev_error):
            if abs(err - prev_error) / max(err, 1e-12) < cfg.stop_rel:
                break
        prev_error = err

        # Enforce positivity on current estimate (x), not previous (xp_prev)
        if cfg.positivity and float(bg) > 0:
            x = xp.where(x < bg, bg, x)

    out = x[:M, :N, :P]
    # Do NOT subtract offset from output - match C reference behavior
    return to_host(out)


__all__ = ["deconvolve_fast", "Backend"]

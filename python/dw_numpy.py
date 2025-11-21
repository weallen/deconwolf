"""
NumPy reference implementation of the Deconwolf RL/SHB algorithms.

This follows the C reference in ``src/method_rl.c`` and ``src/method_shb.c``:
- PSF is normalized, centered, and optionally auto-cropped.
- Optional Anscombe+Gaussian prefiltering of the input/PSF (Van Kempen).
- Work volume padding respects ``border_quality`` (0: periodic, 1: compromise,
  2: full convolution).
- Boundary compensation via Bertero weights for ``border_quality > 0``.
- Iterative Richardson–Lucy or Scaled Heavy Ball (SHB) updates with I-divergence
  or MSE metric.

The intent is to keep this as close as possible to the C code for parity; fast
backends (Numba/JAX/CuPy) can wrap the same logic later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple, List

import numpy as np

Metric = Literal["mse", "idiv"]
Method = Literal["rl", "shb"]
StartCondition = Literal["flat", "identity", "lp"]


# ---------------------- basic math utils ---------------------- #
def anscombe(x: np.ndarray) -> np.ndarray:
    """Anscombe variance-stabilizing transform."""
    return 2.0 * np.sqrt(x + 3.0 / 8.0)


def inverse_anscombe(x: np.ndarray) -> np.ndarray:
    """Closed-form inverse approximation of the Anscombe transform."""
    return (x / 2.0) ** 2 - 3.0 / 8.0


def gaussian_kernel_3d(sx: float, sy: float, sz: float) -> np.ndarray:
    """Create a separable 3D Gaussian kernel with odd support."""
    def _axis_kernel(sigma: float) -> np.ndarray:
        radius = max(1, int(np.ceil(3.0 * sigma)))
        coords = np.arange(-radius, radius + 1, dtype=np.float32)
        kern = np.exp(-0.5 * (coords / sigma) ** 2)
        return kern / kern.sum()

    kx = _axis_kernel(sx)
    ky = _axis_kernel(sy)
    kz = _axis_kernel(sz)
    kernel = np.outer(kx, ky).reshape(kx.size, ky.size, 1) * kz.reshape(1, 1, kz.size)
    return kernel.astype(np.float32)


def fft_convolve_real(x: np.ndarray, kernel: np.ndarray, pad_shape: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """
    Same-size convolution using real FFTs. Kernel is assumed defined near origin;
    it is automatically centered (circshift) to match linear convolution.
    pad_shape can be provided to use a "fast" FFT size (>= x.shape).
    """
    target_shape = pad_shape or x.shape
    work = np.zeros(target_shape, dtype=np.float32)
    insert_shape = tuple(min(a, b) for a, b in zip(kernel.shape, target_shape))
    slices = tuple(slice(0, s) for s in insert_shape)
    work[slices] = kernel[slices]
    work = np.roll(work, shift=tuple(-(np.array(work.shape) // 2)), axis=(0, 1, 2))
    kf = np.fft.rfftn(work, s=target_shape)
    xf = np.fft.rfftn(x, s=target_shape)
    out = np.fft.irfftn(xf * kf, s=target_shape).astype(np.float32)
    return out[(slice(0, x.shape[0]), slice(0, x.shape[1]), slice(0, x.shape[2]))]


def circshift(arr: np.ndarray, shifts: Tuple[int, int, int]) -> np.ndarray:
    return np.roll(arr, shift=shifts, axis=(0, 1, 2))


def max_index(arr: np.ndarray) -> Tuple[int, int, int]:
    return tuple(np.unravel_index(np.nanargmax(arr), arr.shape))


# ---------------------- PSF preparation ---------------------- #
def psf_autocrop_by_image(psf: np.ndarray, im_shape: Tuple[int, int, int], border_quality: int) -> np.ndarray:
    """Mirror of C psf_autocrop_byImage: limit PSF size relative to image and border policy."""
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


def psf_autocrop_xy(psf: np.ndarray, xycropfactor: float) -> np.ndarray:
    """Crop outer x/y slices whose summed mass is below xycropfactor of the max plane sum."""
    m, n, p = psf.shape
    plane_sums = psf.sum(axis=(1, 2))
    maxsum = plane_sums.max()
    if maxsum <= 0:
        return psf

    first = 0
    while first < m and plane_sums[first] < xycropfactor * maxsum:
        first += 1
    if first == 0 or first >= m // 2:
        return psf
    return psf[first : m - first, first : n - first, :]


def psf_autocrop_center_z(psf: np.ndarray) -> np.ndarray:
    """Center PSF along z by keeping symmetric planes around brightest slice."""
    # The C reference does not crop in Z; centering occurs during padding.
    # Keep parity by returning the PSF unchanged.
    return psf


def psf_autocrop(psf: np.ndarray, im_shape: Tuple[int, int, int], border_quality: int, xycropfactor: float) -> np.ndarray:
    psf = psf_autocrop_by_image(psf, im_shape, border_quality)
    if border_quality > 0 and xycropfactor > 0:
        psf = psf_autocrop_xy(psf, xycropfactor)
    # Z-centering is handled when padding/shift to origin; avoid extra cropping here.
    return psf


# ---------------------- core iteration helpers ---------------------- #
def work_shape(im_shape: Tuple[int, int, int], psf_shape: Tuple[int, int, int], border_quality: int) -> Tuple[int, int, int]:
    M, N, P = im_shape
    pM, pN, pP = psf_shape
    if border_quality == 1:
        return (
            int(M + (pM + 1) // 2),
            int(N + (pN + 1) // 2),
            int(P + (pP + 1) // 2),
        )
    if border_quality == 0:
        return max(M, pM), max(N, pN), max(P, pP)
    return M + pM - 1, N + pN - 1, P + pP - 1


def next_fast_shape(shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Pick a potentially faster FFT size using scipy.fft.next_fast_len if available."""
    try:
        from scipy.fft import next_fast_len  # type: ignore
    except ImportError:
        return shape
    return tuple(next_fast_len(s) for s in shape)


def insert_upper_left(target: np.ndarray, source: np.ndarray) -> None:
    """Insert source into the upper-left-front corner of target (in-place)."""
    s0, s1, s2 = source.shape
    target[:s0, :s1, :s2] = source


def center_psf_padded(psf: np.ndarray, wshape: Tuple[int, int, int]) -> np.ndarray:
    """
    Mirror of C centering: pad PSF then circshift so its max lands at (0,0,0).
    """
    Z = np.zeros(wshape, dtype=np.float32)
    insert_upper_left(Z, psf.astype(np.float32))
    maxm, maxn, maxp = max_index(Z)
    Z = circshift(Z, (-maxm, -maxn, -maxp))
    assert Z[0, 0, 0] == Z.max()
    return Z


def centered_psf_fft(psf: np.ndarray, wshape: Tuple[int, int, int]) -> np.ndarray:
    """Place PSF into padded volume, center it, and return real FFT."""
    Z = center_psf_padded(psf, wshape)
    return np.fft.rfftn(Z, s=wshape)


def initial_guess_fft(im_shape: Tuple[int, int, int], wshape: Tuple[int, int, int]) -> np.ndarray:
    """FFT of an indicator image equal to 1 over the observed region."""
    M, N, P = im_shape
    one = np.zeros(wshape, dtype=np.float32)
    one[:M, :N, :P] = 1.0
    return np.fft.rfftn(one, s=wshape)


def compute_weights(cK: np.ndarray, im_shape: Tuple[int, int, int], wshape: Tuple[int, int, int]) -> Optional[np.ndarray]:
    """
    Bertero weights: W = 1 / (K * 1), thresholded at sigma.
    Returns None for border_quality==0 (caller decides).
    """
    F_one = initial_guess_fft(im_shape, wshape)
    W = np.fft.irfftn(np.conj(cK) * F_one, s=wshape).astype(np.float32)
    sigma = 0.01
    mask = W > sigma
    out = np.zeros_like(W, dtype=np.float32)
    out[mask] = 1.0 / W[mask]
    return out


def get_error(y: np.ndarray, g: np.ndarray, metric: Metric) -> float:
    if metric == "mse":
        return get_fmse(y, g)
    return get_idiv(y, g)


def get_fmse(y: np.ndarray, g: np.ndarray) -> float:
    M, N, P = g.shape
    sub = y[:M, :N, :P]
    return float(np.mean((sub - g) ** 2))


def get_idiv(y: np.ndarray, g: np.ndarray) -> float:
    M, N, P = g.shape
    obs = y[:M, :N, :P]
    est = g
    mask = (obs > 0) & (est > 0)
    if not np.any(mask):
        return 0.0
    val = est[mask] * np.log(est[mask] / obs[mask]) - (est[mask] - obs[mask])
    return float(val.sum() / (M * N * P))


# ---------------------- main algorithms ---------------------- #
@dataclass
class DeconvolutionConfig:
    n_iter: int = 10
    alphamax: float = 1.0  # SHB momentum cap (C default is 1.0)
    biggs: bool = False  # Enable Biggs acceleration (RL-style momentum)
    eve: bool = False  # Enable simple Eve-style extrapolation
    biggs_clip: float = 1.0  # max momentum scale
    bg: Optional[float] = None
    psigma: float = 0.0
    border_quality: int = 2
    positivity: bool = True
    xycropfactor: float = 0.001
    metric: Metric = "idiv"
    stop_rel: Optional[float] = None
    stop_abs: Optional[float] = None
    start_condition: StartCondition = "flat"
    start_lpsigma: float = 8.0  # used when start_condition == "lp"
    flatfield: Optional[np.ndarray] = None
    offset: float = 0.0
    zcrop: int = 0  # remove zcrop planes from top and bottom if > 0
    tile_max_size: Optional[int] = None
    tile_overlap: int = 20
    pad_fast_fft: bool = False  # use next-fast-length padding for FFT speed
    fftw_threads: int = 1  # used by fftw backend
    jax_platform: Optional[str] = None  # 'cpu'|'gpu' etc, passed to jax if set
    jax_num_threads: Optional[int] = None  # hint for JAX CPU threads
    use_weights: bool = False  # disable Bertero weights by default (parity with JAX)


def prefilter_im_psf(im: np.ndarray, psf: np.ndarray, psigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Anscombe + Gaussian smoothing and undo Anscombe (C prefilter equivalent)."""
    if psigma <= 0:
        return im.astype(np.float32), psf.astype(np.float32)

    kernel = gaussian_kernel_3d(psigma, psigma, psigma)
    im_f = inverse_anscombe(fft_convolve_real(anscombe(im.astype(np.float32)), kernel))
    psf_f = fft_convolve_real(psf.astype(np.float32), kernel)
    return im_f.astype(np.float32), psf_f.astype(np.float32)


def deconvolve(
    im: np.ndarray,
    psf: np.ndarray,
    method: Method = "shb",
    cfg: Optional[DeconvolutionConfig] = None,
) -> np.ndarray:
    """
    Deconvolve a 3D image with RL or SHB, mirroring the C reference behavior.

    Args:
        im: Input image, float32 or float64, non-negative.
        psf: Point spread function, will be normalized to sum=1 and centered.
        method: 'rl' or 'shb'.
        cfg: Configuration parameters; defaults match the C defaults where applicable.
    """
    if cfg is None:
        cfg = DeconvolutionConfig()
    im = im.astype(np.float32, copy=False)
    psf = psf.astype(np.float32, copy=False)

    if cfg.zcrop > 0:
        if 2 * cfg.zcrop >= im.shape[2]:
            raise ValueError("zcrop too large for image depth")
        im = im[:, :, cfg.zcrop : im.shape[2] - cfg.zcrop]

    if cfg.flatfield is not None:
        # Divide per-plane; flatfield is assumed broadcastable to image
        im = im / cfg.flatfield

    if cfg.offset > 0:
        im = im + cfg.offset

    # Calculate bg AFTER adding offset to match C reference behavior
    if cfg.bg is None:
        bg = max(float(im.min()), 1e-2)
    else:
        bg = float(cfg.bg)

    psf = psf / float(psf.sum())
    psf = psf_autocrop(psf, im.shape, cfg.border_quality, cfg.xycropfactor)
    psf = psf / float(psf.sum())
    im, psf = prefilter_im_psf(im, psf, cfg.psigma)
    psf = psf / float(psf.sum())

    wshape = work_shape(im.shape, psf.shape, cfg.border_quality)
    if cfg.pad_fast_fft:
        wshape = next_fast_shape(wshape)
    cK = centered_psf_fft(psf, wshape)

    W = None
    if cfg.border_quality > 0 and cfg.use_weights:
        W = compute_weights(cK, im.shape, wshape)

    M, N, P = im.shape
    wM, wN, wP = wshape
    sumg = float(im.sum())

    if cfg.start_condition == "flat":
        x = np.full(wshape, sumg / (wM * wN * wP), dtype=np.float32)
    elif cfg.start_condition == "identity":
        x = np.zeros(wshape, dtype=np.float32)
        insert_upper_left(x, im)
    elif cfg.start_condition == "lp":
        lp_im = fft_convolve_real(im, gaussian_kernel_3d(cfg.start_lpsigma, cfg.start_lpsigma, cfg.start_lpsigma),)
        x = np.zeros(wshape, dtype=np.float32)
        insert_upper_left(x, lp_im)
    else:
        raise ValueError(f"Unknown start_condition {cfg.start_condition}")
    xp = x.copy()
    xp_prev2 = xp.copy()

    prev_error = np.inf
    for it in range(cfg.n_iter):
        if method == "shb":
            alpha = max(0.0, min(cfg.alphamax, (it - 1.0) / (it + 2.0)))
            p = x + alpha * (x - xp)
            p[p < bg] = bg
            xp_new, err = iter_shb_step(im, cK, p, W, bg, cfg.metric)
            xp = x  # Save previous iterate for next momentum calculation
            x = xp_new  # advance the iterate
        else:
            current_f = xp
            if cfg.biggs or cfg.eve:
                delta = xp - xp_prev2
                beta = max(0.0, min(cfg.biggs_clip, it / (it + 3.0)))
                if cfg.eve:
                    beta = min(beta, 1.0 - np.exp(-(it + 1)))
                current_f = xp + beta * delta
                current_f[current_f < bg] = bg
            xp_new, err = iter_rl_step(im, cK, current_f, W, bg, cfg.metric)
            xp_prev2 = xp
            xp = xp_new
            x = xp  # current estimate for logging/stop checks

        if cfg.stop_abs is not None and err < cfg.stop_abs:
            break
        if cfg.stop_rel is not None and np.isfinite(prev_error):
            if abs(err - prev_error) / max(err, 1e-12) < cfg.stop_rel:
                break
        prev_error = err

        # Enforce positivity as in the C reference (applies to RL and SHB).
        # Apply to current estimate (x), not previous (xp)
        if cfg.positivity and bg > 0:
            x[x < bg] = bg

    out = xp[:M, :N, :P].copy()
    # Do NOT subtract offset from output - match C reference behavior
    return out


def iter_rl_step(
    im: np.ndarray,
    fft_psf: np.ndarray,
    f: np.ndarray,
    W: Optional[np.ndarray],
    bg: float,
    metric: Metric,
) -> Tuple[np.ndarray, float]:
    """One RL iteration (C: iter_rl)."""
    M, N, P = im.shape
    wshape = f.shape
    F = np.fft.rfftn(f, s=wshape)
    y = np.fft.irfftn(fft_psf * F, s=wshape).astype(np.float32)
    error = get_error(y, im, metric)

    y_obs = y[:M, :N, :P]
    # Match JAX behavior: replace bad divisors with bg before ratio.
    y_safe = np.where(y_obs > 0, y_obs, bg)
    ratio = im / y_safe

    # Outside the observed region should be a small positive constant, not zero
    # (C fills with 1e-6) to avoid killing gradients at the borders.
    y_full = np.full_like(y, 1e-6, dtype=np.float32)
    y_full[:M, :N, :P] = ratio

    F_sn = np.fft.rfftn(y_full, s=wshape)
    x = np.fft.irfftn(np.conj(fft_psf) * F_sn, s=wshape).astype(np.float32)
    if W is not None:
        x *= f * W
    else:
        x *= f
    return x, error


def iter_shb_step(
    im: np.ndarray,
    cK: np.ndarray,
    pK: np.ndarray,
    W: Optional[np.ndarray],
    bg: float,
    metric: Metric,
) -> Tuple[np.ndarray, float]:
    """One SHB iteration (C: iter_shb)."""
    M, N, P = im.shape
    wshape = pK.shape
    Pk = np.fft.rfftn(pK, s=wshape)
    # Forward model should be a convolution with the PSF (no conjugate), same as C SHB
    y = np.fft.irfftn(cK * Pk, s=wshape).astype(np.float32)
    error = get_error(y, im, metric)

    mindiv = 1e-6
    y_obs = y[:M, :N, :P]
    # Match C reference: clamp small magnitudes but keep the sign to avoid zero divisors
    y_safe = np.where(np.abs(y_obs) < mindiv, np.copysign(mindiv, y_obs), y_obs)
    ratio = im / y_safe

    # Mirror the C reference: outside the observed region is set to zero.
    y_full = np.zeros_like(y, dtype=np.float32)
    y_full[:M, :N, :P] = ratio

    Y = np.fft.rfftn(y_full, s=wshape)
    x = np.fft.irfftn(np.conj(cK) * Y, s=wshape).astype(np.float32)
    if W is not None:
        x *= pK * W
    else:
        x *= pK
    return x, error


__all__ = [
    "DeconvolutionConfig",
    "deconvolve",
    "deconvolve_tiled",
    "iter_rl_step",
    "iter_shb_step",
    "psf_autocrop",
    "work_shape",
]


# ---------------------- tiling support ---------------------- #
def _divide_axis(length: int, max_size: int) -> List[Tuple[int, int]]:
    """Divide an axis into contiguous segments of size <= max_size."""
    spans: List[Tuple[int, int]] = []
    start = 0
    while start < length:
        end = min(length, start + max_size)
        spans.append((start, end))
        start = end
    return spans


def _tile_weight_1d(a: int, b: int, c: int, d: int, x: int) -> float:
    # Mirror of getWeight1d in tiling.c
    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a <= x < b:
        return (x - a + 1) / (b - a + 1)
    if c < x <= d:
        return 1.0 - (x - c) / (d - c + 1)
    return 0.0


def _tile_weight(tile_pos: Tuple[int, int, int, int], x_size: Tuple[int, int, int], idx: Tuple[int, int, int]) -> float:
    # tile_pos: (m0,m1,n0,n1) without padding, x_size: (xm0,xm1,xn0,xn1) with padding
    xm0, xm1, xn0, xn1 = x_size
    m0, m1, n0, n1 = tile_pos
    m, n, _ = idx
    wm = _tile_weight_1d(xm0, m0, m1, xm1, m)
    wn = _tile_weight_1d(xn0, n0, n1, xn1, n)
    return min(wm, wn)

def _tile_weight_grid(xm0, xm1, m0, m1, xn0, xn1, n0, n1, P):
    """
    Vectorized weight grid for a tile with padded extents [xm0,xm1],[xn0,xn1].
    """
    m_coords = np.arange(xm0, xm1 + 1)
    n_coords = np.arange(xn0, xn1 + 1)
    wm = np.ones_like(m_coords, dtype=np.float32)
    wn = np.ones_like(n_coords, dtype=np.float32)

    # leading ramp
    if m0 > xm0:
        ramp_len = m0 - xm0 + 1
        wm[:ramp_len] = (m_coords[:ramp_len] - xm0 + 1) / ramp_len
    # trailing ramp
    if xm1 > m1:
        ramp_len = xm1 - m1 + 1
        wm[-ramp_len:] = 1.0 - (m_coords[-ramp_len:] - m1) / ramp_len

    if n0 > xn0:
        ramp_len = n0 - xn0 + 1
        wn[:ramp_len] = (n_coords[:ramp_len] - xn0 + 1) / ramp_len
    if xn1 > n1:
        ramp_len = xn1 - n1 + 1
        wn[-ramp_len:] = 1.0 - (n_coords[-ramp_len:] - n1) / ramp_len

    weight_2d = np.minimum(wm[:, None], wn[None, :]).astype(np.float32)
    if P > 1:
        return np.repeat(weight_2d[:, :, None], P, axis=2)
    return weight_2d[:, :, None]


def deconvolve_tiled(
    im: np.ndarray,
    psf: np.ndarray,
    method: Method = "shb",
    cfg: Optional[DeconvolutionConfig] = None,
) -> np.ndarray:
    """Process image in tiles with weighted blending to reduce seams."""
    cfg = cfg or DeconvolutionConfig()
    if cfg.tile_max_size is None:
        return deconvolve(im, psf, method=method, cfg=cfg)

    tile_size = cfg.tile_max_size
    overlap = cfg.tile_overlap
    M, N, P = im.shape
    out = np.zeros_like(im, dtype=np.float32)
    weights = np.zeros_like(im, dtype=np.float32)

    m_spans = _divide_axis(M, tile_size)
    n_spans = _divide_axis(N, tile_size)

    for m0, m1 in m_spans:
        for n0, n1 in n_spans:
            xm0 = max(0, m0 - overlap)
            xm1 = min(M - 1, m1 + overlap - 1)
            xn0 = max(0, n0 - overlap)
            xn1 = min(N - 1, n1 + overlap - 1)

            tile_im = im[xm0 : xm1 + 1, xn0 : xn1 + 1, :]
            tile_cfg = DeconvolutionConfig(
                **{**cfg.__dict__, "tile_max_size": None},
            )
            tile_out = deconvolve(tile_im, psf, method=method, cfg=tile_cfg)

            wgrid = _tile_weight_grid(xm0, xm1, m0, m1 - 1, xn0, xn1, n0, n1 - 1, P)
            roi_out = out[xm0 : xm1 + 1, xn0 : xn1 + 1, :]
            roi_w = weights[xm0 : xm1 + 1, xn0 : xn1 + 1, :]
            roi_out += wgrid * tile_out
            roi_w += wgrid

    weights[weights == 0] = 1.0
    return out / weights

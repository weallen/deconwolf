#!/usr/bin/env python3
"""
Benchmark scipy multi-threaded FFT vs JAX for deconvolution.
"""

import numpy as np
import time
import sys

sys.path.insert(0, '/Users/wea/src/allenlab/deconwolf')

from dwpy.dw_jax import _optimal_fft_size
from scipy.fft import rfftn, irfftn, set_workers

N_ITER = 10


def create_test_data(size_xy, size_z, psf_size=31):
    """Create synthetic test data."""
    np.random.seed(42)
    im = np.random.rand(size_xy, size_xy, size_z).astype(np.float32) * 100 + 100
    for _ in range(50):
        x, y = np.random.randint(20, size_xy-20, 2)
        z = np.random.randint(0, size_z)
        im[x-5:x+5, y-5:y+5, z] += 500

    psf = np.zeros((psf_size, psf_size, psf_size), dtype=np.float32)
    center = psf_size // 2
    for i in range(psf_size):
        for j in range(psf_size):
            for k in range(psf_size):
                r2 = (i-center)**2 + (j-center)**2 + (k-center)**2
                psf[i,j,k] = np.exp(-r2 / (2 * 5**2))
    psf /= psf.sum()
    return im, psf


def scipy_decon_shb(im, psf, n_iter, border_quality, n_workers=-1):
    """SHB deconvolution using scipy FFT with multi-threading."""
    M, N, P = im.shape
    pM, pN, pP = psf.shape

    if border_quality == 2:
        wM, wN, wP = M + pM - 1, N + pN - 1, P + pP - 1
    elif border_quality == 1:
        wM, wN, wP = M + (pM + 1)//2, N + (pN + 1)//2, P + (pP + 1)//2
    else:
        wM, wN, wP = max(M, pM), max(N, pN), max(P, pP)

    # Optimize FFT sizes
    wM = _optimal_fft_size(wM)
    wN = _optimal_fft_size(wN)
    wP = _optimal_fft_size(wP)
    wshape = (wM, wN, wP)

    bg = max(im.min(), 1e-2)

    # Prepare PSF
    psf = psf / psf.sum()
    Z = np.zeros(wshape, dtype=np.float32)
    Z[:pM, :pN, :pP] = psf
    max_idx = np.unravel_index(np.argmax(Z), Z.shape)
    Z = np.roll(Z, [-max_idx[0], -max_idx[1], -max_idx[2]], axis=[0, 1, 2])

    with set_workers(n_workers):
        cK = rfftn(Z)

    # Initial guess
    x = np.full(wshape, im.sum() / (wM * wN * wP), dtype=np.float32)
    xp = x.copy()
    mindiv = 1e-6
    alphamax = 10.0

    for i in range(n_iter):
        alpha = min(max((i - 1.0) / (i + 2.0), 0.0), alphamax)
        p = x + alpha * (x - xp)
        p = np.maximum(p, bg)

        with set_workers(n_workers):
            pK = rfftn(p)
            y = irfftn(cK * pK, s=wshape)

        y_obs = y[:M, :N, :P]
        y_safe = np.where(np.abs(y_obs) < mindiv, np.copysign(mindiv, y_obs), y_obs)
        ratio = im / y_safe

        y_full = np.zeros(wshape, dtype=np.float32)
        y_full[:M, :N, :P] = ratio

        with set_workers(n_workers):
            Y = rfftn(y_full)
            x_new = irfftn(np.conj(cK) * Y, s=wshape)

        x_new = x_new * p
        xp = x
        x = x_new

    return x[:M, :N, :P]


def main():
    print("=" * 60)
    print("SciPy Multi-threaded FFT Benchmark (2048x2048x32)")
    print("=" * 60)
    print(f"Iterations: {N_ITER}")
    print()

    im, psf = create_test_data(2048, 32)
    print(f"Image shape: {im.shape}, PSF shape: {psf.shape}")
    print()

    # Test with border_quality=0 (same as best JAX config)
    print("Testing scipy with border_quality=0:")
    print("-" * 60)

    for n_workers in [1, 2, 4, 8, -1]:
        worker_str = "all" if n_workers == -1 else str(n_workers)
        print(f"  workers={worker_str}...", end='', flush=True)

        # Warmup
        _ = scipy_decon_shb(im, psf, n_iter=2, border_quality=0, n_workers=n_workers)

        # Timed run
        start = time.time()
        result = scipy_decon_shb(im, psf, n_iter=N_ITER, border_quality=0, n_workers=n_workers)
        elapsed = time.time() - start
        print(f" {elapsed:.1f}s")

    print()
    print("Reference: JAX bq=0 fft_opt=True = 31.0s")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Comprehensive benchmark for JAX deconvolution optimizations on 2048x2048 images.

Tests:
1. border_quality: 0, 1, 2
2. optimize_fft_size: True/False
3. scipy with multi-threading vs JAX
"""

import numpy as np
import jax
import jax.numpy as jnp
import time
import sys
import os

# Enable multi-threading for scipy
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'

sys.path.insert(0, '/Users/wea/src/allenlab/deconwolf')

from dwpy.dw_jax import run_dw, _optimal_fft_size
from scipy import ndimage
from scipy.fft import rfftn, irfftn, set_workers

N_ITER = 10


def create_test_data(size_xy, size_z, psf_size=31):
    """Create synthetic test data."""
    np.random.seed(42)

    im = np.random.rand(size_xy, size_xy, size_z).astype(np.float32) * 100 + 100

    # Add bright spots
    for _ in range(50):
        x, y = np.random.randint(20, size_xy-20, 2)
        z = np.random.randint(0, size_z)
        im[x-5:x+5, y-5:y+5, z] += 500

    # Create PSF (Gaussian-like)
    psf = np.zeros((psf_size, psf_size, psf_size), dtype=np.float32)
    center = psf_size // 2
    for i in range(psf_size):
        for j in range(psf_size):
            for k in range(psf_size):
                r2 = (i-center)**2 + (j-center)**2 + (k-center)**2
                psf[i,j,k] = np.exp(-r2 / (2 * 5**2))
    psf /= psf.sum()

    return im, psf


def benchmark_jax(im, psf, border_quality, optimize_fft_size):
    """Benchmark JAX deconvolution."""
    im_jax = jnp.array(im)
    psf_jax = jnp.array(psf)

    # Warmup
    _ = run_dw(im_jax, psf_jax, n_iter=N_ITER, alphamax=10.0, verbose=False,
               err_thresh=None, border_quality=border_quality,
               optimize_fft_size=optimize_fft_size)
    jax.block_until_ready(_)

    # Timed run
    start = time.time()
    result = run_dw(im_jax, psf_jax, n_iter=N_ITER, alphamax=10.0, verbose=False,
                    err_thresh=None, border_quality=border_quality,
                    optimize_fft_size=optimize_fft_size)
    jax.block_until_ready(result)
    elapsed = time.time() - start

    return elapsed, result


def scipy_decon_shb(im, psf, n_iter, border_quality, bg=None, n_workers=-1):
    """SHB deconvolution using scipy FFT with multi-threading."""
    M, N, P = im.shape
    pM, pN, pP = psf.shape

    # Compute working shape based on border_quality
    if border_quality == 2:
        wM, wN, wP = M + pM - 1, N + pN - 1, P + pP - 1
    elif border_quality == 1:
        wM, wN, wP = M + (pM + 1)//2, N + (pN + 1)//2, P + (pP + 1)//2
    else:  # border_quality == 0
        wM, wN, wP = max(M, pM), max(N, pN), max(P, pP)

    # Optimize FFT sizes
    wM = _optimal_fft_size(wM)
    wN = _optimal_fft_size(wN)
    wP = _optimal_fft_size(wP)

    wshape = (wM, wN, wP)

    if bg is None:
        bg = max(im.min(), 1e-2)

    # Prepare PSF
    psf = psf / psf.sum()
    Z = np.zeros(wshape, dtype=np.float32)
    Z[:pM, :pN, :pP] = psf

    # Center PSF
    max_idx = np.unravel_index(np.argmax(Z), Z.shape)
    Z = np.roll(Z, [-max_idx[0], -max_idx[1], -max_idx[2]], axis=[0, 1, 2])

    # PSF FFT
    with set_workers(n_workers):
        cK = rfftn(Z)

    # Initial guess
    x = np.full(wshape, im.sum() / (wM * wN * wP), dtype=np.float32)
    xp = x.copy()

    mindiv = 1e-6
    alphamax = 10.0

    for i in range(n_iter):
        # Momentum
        alpha = min(max((i - 1.0) / (i + 2.0), 0.0), alphamax)
        p = x + alpha * (x - xp)
        p = np.maximum(p, bg)

        # Forward model
        with set_workers(n_workers):
            pK = rfftn(p)
            y = irfftn(cK * pK, s=wshape)

        # Compute ratio
        y_obs = y[:M, :N, :P]
        y_safe = np.where(np.abs(y_obs) < mindiv, np.copysign(mindiv, y_obs), y_obs)
        ratio = im / y_safe

        # Back-project
        y_full = np.zeros(wshape, dtype=np.float32)
        y_full[:M, :N, :P] = ratio

        with set_workers(n_workers):
            Y = rfftn(y_full)
            x_new = irfftn(np.conj(cK) * Y, s=wshape)

        x_new = x_new * p

        xp = x
        x = x_new

    return x[:M, :N, :P]


def benchmark_scipy(im, psf, border_quality, n_workers):
    """Benchmark scipy deconvolution."""
    # Warmup
    _ = scipy_decon_shb(im, psf, n_iter=2, border_quality=border_quality, n_workers=n_workers)

    # Timed run
    start = time.time()
    result = scipy_decon_shb(im, psf, n_iter=N_ITER, border_quality=border_quality, n_workers=n_workers)
    elapsed = time.time() - start

    return elapsed, result


def compute_correlation(a, b):
    """Compute correlation between two arrays."""
    a_flat = np.array(a).flatten()
    b_flat = np.array(b).flatten()
    return np.corrcoef(a_flat, b_flat)[0, 1]


def main():
    print("=" * 70)
    print("Comprehensive Optimization Benchmark for 2048x2048x32 Deconvolution")
    print("=" * 70)
    print(f"Iterations: {N_ITER}")
    print(f"JAX devices: {jax.devices()}")
    print()

    # Create test data
    print("Creating test data (2048x2048x32)...")
    im, psf = create_test_data(2048, 32)
    print(f"Image shape: {im.shape}, PSF shape: {psf.shape}")

    # Show what FFT sizes will be used
    print("\nFFT working shapes by border_quality (with optimization):")
    M, N, P = im.shape
    pM, pN, pP = psf.shape
    for bq in [0, 1, 2]:
        if bq == 2:
            wM, wN, wP = M + pM - 1, N + pN - 1, P + pP - 1
        elif bq == 1:
            wM, wN, wP = M + (pM + 1)//2, N + (pN + 1)//2, P + (pP + 1)//2
        else:
            wM, wN, wP = max(M, pM), max(N, pN), max(P, pP)
        wM_opt = _optimal_fft_size(wM)
        wN_opt = _optimal_fft_size(wN)
        wP_opt = _optimal_fft_size(wP)
        print(f"  bq={bq}: raw ({wM}x{wN}x{wP}) -> optimized ({wM_opt}x{wN_opt}x{wP_opt})")

    results = {}

    print("\n" + "-" * 70)
    print("JAX Tests:")
    print("-" * 70)

    # Test JAX with different settings
    for bq in [0, 1, 2]:
        for opt_fft in [True, False]:
            name = f"JAX bq={bq} fft_opt={opt_fft}"
            print(f"  {name}...", end='', flush=True)
            elapsed, result = benchmark_jax(im, psf, bq, opt_fft)
            results[name] = {'time': elapsed, 'result': result}
            print(f" {elapsed:.1f}s")

    print("\n" + "-" * 70)
    print("SciPy Multi-threaded Tests:")
    print("-" * 70)

    # Test scipy with different thread counts
    for n_workers in [1, 4, 8, -1]:  # -1 means all available
        worker_str = "all" if n_workers == -1 else str(n_workers)
        name = f"SciPy bq=1 workers={worker_str}"
        print(f"  {name}...", end='', flush=True)
        elapsed, result = benchmark_scipy(im, psf, border_quality=1, n_workers=n_workers)
        results[name] = {'time': elapsed, 'result': result}
        print(f" {elapsed:.1f}s")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<35} {'Time (s)':<12} {'Speedup':<10}")
    print("-" * 70)

    # Use JAX bq=2 as baseline (original default)
    baseline_time = results['JAX bq=2 fft_opt=True']['time']
    baseline_result = results['JAX bq=2 fft_opt=True']['result']

    for key, data in sorted(results.items(), key=lambda x: x[1]['time']):
        speedup = baseline_time / data['time']
        corr = compute_correlation(baseline_result, data['result'])
        print(f"{key:<35} {data['time']:<12.1f} {speedup:<10.2f}x (corr: {corr:.4f})")

    # Find best overall
    print("\n" + "-" * 70)
    best = min(results.items(), key=lambda x: x[1]['time'])
    print(f"Best: {best[0]} ({best[1]['time']:.1f}s)")
    print(f"Speedup vs default (JAX bq=2): {baseline_time / best[1]['time']:.2f}x")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Benchmark different tile sizes for parallel tiled deconvolution on 2048x2048 images.

Compares:
1. Full image JAX (run_dw) - no tiling
2. Sequential tiled (run_dw_tiled)
3. Parallel tiled (run_dw_tiled_parallel) with various tile sizes
"""

import numpy as np
import jax
import jax.numpy as jnp
import time
import sys

sys.path.insert(0, '/Users/wea/src/allenlab/deconwolf')

from dwpy.dw_jax import run_dw, run_dw_tiled, run_dw_tiled_parallel, decon_fast

# Reduce iterations for faster benchmarking
N_ITER = 10
N_RUNS = 1  # Single run for quick comparison


def create_test_data(size_xy, size_z, psf_size=31):
    """Create synthetic test data."""
    np.random.seed(42)

    # Create image with some structure
    im = np.random.rand(size_xy, size_xy, size_z).astype(np.float32) * 100 + 100

    # Add some bright spots
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

    return jnp.array(im), jnp.array(psf)


def benchmark_method(name, func, im, psf, **kwargs):
    """Benchmark a deconvolution method."""
    times = []

    # Warmup run
    print(f"  {name}: warming up...", end='', flush=True)
    _ = func(im, psf, **kwargs)
    jax.block_until_ready(_)
    print(" done. Running...", end='', flush=True)

    # Timed runs
    for i in range(N_RUNS):
        start = time.time()
        result = func(im, psf, **kwargs)
        jax.block_until_ready(result)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f" {elapsed:.1f}s", end='', flush=True)

    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f" -> avg: {avg_time:.2f}s ± {std_time:.2f}s")

    return result, avg_time


def compute_correlation(a, b):
    """Compute correlation between two arrays."""
    a_flat = np.array(a).flatten()
    b_flat = np.array(b).flatten()
    return np.corrcoef(a_flat, b_flat)[0, 1]


def main():
    print("=" * 70)
    print("Tile Size Benchmark for 2048x2048x32 Deconvolution")
    print("=" * 70)
    print(f"Iterations: {N_ITER}")
    print(f"Runs per method: {N_RUNS}")
    print(f"JAX devices: {jax.devices()}")
    print()

    # Create test data
    print("Creating test data (2048x2048x32)...")
    im, psf = create_test_data(2048, 32)
    print(f"Image shape: {im.shape}, PSF shape: {psf.shape}")
    print()

    results = {}

    # Test different tile sizes (skip 128 - too many tiles causes memory issues)
    tile_sizes = [256, 512]
    tile_padding = 40

    print("-" * 70)
    print("Parallel Tiled (run_dw_tiled_parallel):")
    print("-" * 70)

    for tile_size in tile_sizes:
        n_tiles_approx = (1024 // tile_size) ** 2
        name = f"Tiled Parallel {tile_size}x{tile_size} (~{n_tiles_approx} tiles)"
        result, time_taken = benchmark_method(
            name, run_dw_tiled_parallel, im, psf,
            tile_max_size=tile_size, tile_padding=tile_padding,
            n_iter=N_ITER, alphamax=10.0
        )
        results[f'parallel_{tile_size}'] = {'time': time_taken, 'result': result}

    print()
    print("-" * 70)
    print("Full Image (no tiling):")
    print("-" * 70)

    # Full image with run_dw
    name = "Full Image (run_dw)"
    result, time_taken = benchmark_method(
        name, run_dw, im, psf,
        n_iter=N_ITER, alphamax=10.0, verbose=False, err_thresh=None
    )
    results['full_original'] = {'time': time_taken, 'result': result}

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<45} {'Time (s)':<12} {'Speedup':<10}")
    print("-" * 70)

    baseline_time = results['full_original']['time']
    baseline_result = results['full_original']['result']

    for key, data in sorted(results.items(), key=lambda x: x[1]['time']):
        speedup = baseline_time / data['time']
        corr = compute_correlation(baseline_result, data['result'])
        print(f"{key:<45} {data['time']:<12.2f} {speedup:<10.2f}x (corr: {corr:.6f})")

    # Find best parallel tile size
    print()
    print("-" * 70)
    parallel_results = {k: v for k, v in results.items() if k.startswith('parallel_')}
    if parallel_results:
        best = min(parallel_results.items(), key=lambda x: x[1]['time'])
        print(f"Best parallel tile size: {best[0]} ({best[1]['time']:.2f}s)")
        print(f"Speedup vs full image: {baseline_time / best[1]['time']:.2f}x")


if __name__ == '__main__':
    main()

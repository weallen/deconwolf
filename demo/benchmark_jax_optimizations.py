"""
Benchmark script for JAX deconvolution optimizations.

Compares:
1. C implementation (reference)
2. Original JAX implementation (Python loop with per-iteration JIT)
3. New optimized JAX (jax.lax.fori_loop - single JIT compilation)

Tests on synthetic data with various sizes to measure speedup.
"""
import sys
from pathlib import Path
import time
import numpy as np
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Check JAX availability
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
except ImportError:
    JAX_AVAILABLE = False
    print("JAX not available, exiting")
    sys.exit(1)

# Check for C executable
DW_C_PATH = Path("/Users/wea/src/deconwolf/build/dw")
C_AVAILABLE = DW_C_PATH.exists()
print(f"C executable: {'Available' if C_AVAILABLE else 'Not found'}")


def create_synthetic_data(size=(128, 128, 32), psf_size=(31, 31, 15)):
    """Create synthetic test data."""
    np.random.seed(42)

    # Create image with some structure
    im = np.random.rand(*size).astype(np.float32) * 100 + 100

    # Add some bright spots
    for _ in range(10):
        x = np.random.randint(10, size[0]-10)
        y = np.random.randint(10, size[1]-10)
        z = np.random.randint(5, size[2]-5)
        im[x-3:x+3, y-3:y+3, z-2:z+2] += 500

    # Create Gaussian PSF
    px, py, pz = psf_size
    cx, cy, cz = px//2, py//2, pz//2
    x = np.arange(px) - cx
    y = np.arange(py) - cy
    z = np.arange(pz) - cz
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    sigma_xy = 3.0
    sigma_z = 2.0
    psf = np.exp(-(X**2 + Y**2)/(2*sigma_xy**2) - Z**2/(2*sigma_z**2))
    psf = psf.astype(np.float32)
    psf /= psf.sum()

    return im, psf


def preprocess_for_decon(im, psf):
    """Apply same preprocessing as run_dw() for fair comparison."""
    from dwpy.dw_jax import psf_autocrop

    # Image normalization (same as run_dw)
    im = im.copy()
    if im.min() < 0:
        im -= im.min()
    if im.max() < 1000:
        im = im * (1000 / im.max())

    # PSF normalization and autocrop (same as run_dw)
    psf = psf.copy()
    psf = psf / psf.sum()
    psf = psf_autocrop(psf, im)
    psf = psf / psf.sum()

    return im, psf


def benchmark_c(im, psf, n_iter=50, n_runs=1):
    """Benchmark C implementation."""
    import tifffile as tf

    if not C_AVAILABLE:
        return None, None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        im_path = tmpdir / "input.tif"
        psf_path = tmpdir / "psf.tif"
        out_path = tmpdir / "output.tif"

        # Save input files (C expects ZYX order)
        im_zyx = np.transpose(im, (2, 1, 0))
        psf_zyx = np.transpose(psf, (2, 1, 0))

        # Center PSF for C compatibility
        max_idx = np.unravel_index(np.argmax(psf_zyx), psf_zyx.shape)
        psf_centered = np.roll(psf_zyx, -max_idx[0], axis=0)
        psf_centered = np.roll(psf_centered, -max_idx[1], axis=1)
        psf_centered = np.roll(psf_centered, -max_idx[2], axis=2)
        psf_centered = psf_centered / psf_centered.sum()

        tf.imwrite(im_path, im_zyx.astype(np.float32))
        tf.imwrite(psf_path, psf_centered.astype(np.float32))

        times = []
        for run in range(n_runs):
            start = time.perf_counter()
            result = subprocess.run(
                [str(DW_C_PATH), "--iter", str(n_iter),
                 str(im_path), str(psf_path),
                 "--overwrite", "--noplan", "--float",
                 "--method", "shb",
                 "-o", str(out_path)],
                capture_output=True,
                text=True,
                timeout=600
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            if n_runs > 1:
                print(f"    Run {run+1}/{n_runs}: {elapsed:.3f}s")

        # Load result
        if out_path.exists():
            result_zyx = tf.imread(out_path).astype(np.float32)
            result_xyz = np.transpose(result_zyx, (2, 1, 0))
            return result_xyz, times
        else:
            return None, times


def benchmark_original_jax(im, psf, n_iter=50, n_runs=3):
    """Benchmark original JAX implementation using decon() directly."""
    from dwpy.dw_jax import decon

    # Apply same preprocessing
    im_proc, psf_proc = preprocess_for_decon(im, psf)
    im_jax = jnp.array(im_proc)
    psf_jax = jnp.array(psf_proc)

    # Warmup
    print("  Warming up original JAX (decon)...", flush=True)
    _ = decon(im_jax, psf_jax, psigma=0, n_iter=2, verbose=False, err_thresh=None, method='shb_jit')
    jax.block_until_ready(_)

    # Benchmark
    times = []
    for i in range(n_runs):
        start = time.perf_counter()
        result = decon(im_jax, psf_jax, psigma=0, n_iter=n_iter, verbose=False, err_thresh=None, method='shb_jit')
        jax.block_until_ready(result)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"    Run {i+1}/{n_runs}: {elapsed:.3f}s")

    return np.array(result), times


def benchmark_optimized_jax(im, psf, n_iter=50, n_runs=3):
    """Benchmark new optimized JAX implementation (fori_loop)."""
    from dwpy.dw_jax import decon_fast

    # Apply same preprocessing
    im_proc, psf_proc = preprocess_for_decon(im, psf)
    im_jax = jnp.array(im_proc)
    psf_jax = jnp.array(psf_proc)

    # Warmup (includes JIT compilation)
    print("  Warming up optimized JAX (fori_loop)...", flush=True)
    _ = decon_fast(im_jax, psf_jax, psigma=0, n_iter=2)
    jax.block_until_ready(_)

    # Benchmark
    times = []
    for i in range(n_runs):
        start = time.perf_counter()
        result = decon_fast(im_jax, psf_jax, psigma=0, n_iter=n_iter)
        jax.block_until_ready(result)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"    Run {i+1}/{n_runs}: {elapsed:.3f}s")

    return np.array(result), times


def benchmark_batch_jax(images, psf, n_iter=50, n_runs=3):
    """Benchmark batch processing with vmap."""
    from dwpy.dw_jax import batch_deconvolve

    # Preprocess PSF (same for all images in batch)
    _, psf_proc = preprocess_for_decon(images[0], psf)
    psf_jax = jnp.array(psf_proc)

    # Preprocess each image in batch
    images_proc = []
    for img in images:
        im_proc, _ = preprocess_for_decon(img, psf)
        images_proc.append(im_proc)
    images_jax = jnp.array(np.stack(images_proc))

    # Warmup
    print("  Warming up batch JAX (vmap)...", flush=True)
    _ = batch_deconvolve(images_jax[:1], psf_jax, n_iter=2)
    jax.block_until_ready(_)

    # Benchmark
    times = []
    for i in range(n_runs):
        start = time.perf_counter()
        result = batch_deconvolve(images_jax, psf_jax, n_iter=n_iter)
        jax.block_until_ready(result)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"    Run {i+1}/{n_runs}: {elapsed:.3f}s")

    return np.array(result), times


def calculate_correlation(a, b):
    """Calculate Pearson correlation coefficient."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    return np.corrcoef(a_flat, b_flat)[0, 1]


def main():
    print("="*70)
    print("JAX Deconvolution Optimization Benchmark")
    print("="*70)

    # Test configurations: (image_size, psf_size, n_iter)
    test_sizes = [
        ((128, 128, 32), (31, 31, 15), 50),    # Medium
        ((256, 256, 32), (31, 31, 15), 50),    # Large
        ((512, 512, 32), (31, 31, 15), 50),    # XL
        ((1024, 1024, 32), (31, 31, 15), 50),  # XXL
        ((2048, 2048, 32), (31, 31, 15), 50),  # Huge
    ]

    results = {}

    for im_size, psf_size, n_iter in test_sizes:
        print(f"\n{'='*70}")
        print(f"Test: Image {im_size}, PSF {psf_size}, {n_iter} iterations")
        print("="*70)

        im, psf = create_synthetic_data(im_size, psf_size)
        n_voxels = im_size[0] * im_size[1] * im_size[2]
        print(f"Image: {im.shape} ({n_voxels:,} voxels)")

        size_results = {'n_voxels': n_voxels, 'n_iter': n_iter}

        # Benchmark C
        if C_AVAILABLE:
            print("\n1. C (FFTW):")
            try:
                result_c, times_c = benchmark_c(im, psf, n_iter=n_iter, n_runs=1)
                if times_c:
                    mean_c = np.mean(times_c)
                    print(f"   Time: {mean_c:.3f}s ({mean_c/n_iter*1000:.1f}ms/iter)")
                    size_results['c'] = mean_c
                else:
                    size_results['c'] = None
            except Exception as e:
                print(f"   FAILED: {e}")
                size_results['c'] = None
        else:
            print("\n1. C (FFTW): Not available")
            size_results['c'] = None

        # Benchmark original JAX
        print("\n2. JAX Original (Python loop):")
        try:
            result_orig, times_orig = benchmark_original_jax(im, psf, n_iter=n_iter, n_runs=1)
            if times_orig:
                mean_orig = np.mean(times_orig)
                print(f"   Time: {mean_orig:.3f}s ({mean_orig/n_iter*1000:.1f}ms/iter)")
                size_results['jax_orig'] = mean_orig
            else:
                size_results['jax_orig'] = None
        except Exception as e:
            print(f"   FAILED: {e}")
            import traceback
            traceback.print_exc()
            size_results['jax_orig'] = None
            result_orig = None

        # Benchmark optimized JAX
        print("\n3. JAX Optimized (fori_loop):")
        try:
            result_opt, times_opt = benchmark_optimized_jax(im, psf, n_iter=n_iter, n_runs=1)
            if times_opt:
                mean_opt = np.mean(times_opt)
                print(f"   Time: {mean_opt:.3f}s ({mean_opt/n_iter*1000:.1f}ms/iter)")
                size_results['jax_opt'] = mean_opt

                if result_orig is not None:
                    corr = calculate_correlation(result_orig, result_opt)
                    print(f"   Correlation: {corr:.6f}")
            else:
                size_results['jax_opt'] = None
        except Exception as e:
            print(f"   FAILED: {e}")
            import traceback
            traceback.print_exc()
            size_results['jax_opt'] = None

        # Summary for this size
        print(f"\n   --- Summary for {im_size} ---")
        if size_results.get('c'):
            print(f"   C:            {size_results['c']:.3f}s")
        if size_results.get('jax_orig'):
            print(f"   JAX Original: {size_results['jax_orig']:.3f}s", end="")
            if size_results.get('c'):
                print(f" ({size_results['jax_orig']/size_results['c']:.2f}x vs C)")
            else:
                print()
        if size_results.get('jax_opt'):
            print(f"   JAX Optimized:{size_results['jax_opt']:.3f}s", end="")
            if size_results.get('jax_orig'):
                speedup = size_results['jax_orig'] / size_results['jax_opt']
                print(f" ({speedup:.2f}x vs JAX Orig)", end="")
            if size_results.get('c'):
                print(f" ({size_results['jax_opt']/size_results['c']:.2f}x vs C)")
            else:
                print()

        results[str(im_size)] = size_results

    # Batch processing test
    print(f"\n{'='*70}")
    print("Batch Processing Test (4 images, 128x128x32)")
    print("="*70)

    im, psf = create_synthetic_data((128, 128, 32), (31, 31, 15))
    batch_size = 4
    images = np.stack([im + np.random.rand(*im.shape).astype(np.float32) * 10
                       for _ in range(batch_size)])
    print(f"Batch shape: {images.shape}")

    # Preprocess for fair comparison
    _, psf_proc = preprocess_for_decon(images[0], psf)
    psf_jax = jnp.array(psf_proc)
    images_proc = []
    for img in images:
        im_proc, _ = preprocess_for_decon(img, psf)
        images_proc.append(im_proc)
    images_jax = jnp.array(np.stack(images_proc))

    # Sequential (process one at a time with optimized)
    print("\n1. Sequential processing (optimized, one at a time):")
    from dwpy.dw_jax import decon_fast

    # Warmup
    _ = decon_fast(images_jax[0], psf_jax, psigma=0, n_iter=2)
    jax.block_until_ready(_)

    times_seq = []
    for run in range(3):
        start = time.perf_counter()
        results_seq = []
        for i in range(batch_size):
            r = decon_fast(images_jax[i], psf_jax, psigma=0, n_iter=50)
            results_seq.append(r)
        for r in results_seq:
            jax.block_until_ready(r)
        elapsed = time.perf_counter() - start
        times_seq.append(elapsed)
        print(f"    Run {run+1}/3: {elapsed:.3f}s")

    mean_seq = np.mean(times_seq)
    print(f"   Average: {mean_seq:.3f}s")

    # Batch with vmap
    print("\n2. Batch processing (vmap):")
    try:
        result_batch, times_batch = benchmark_batch_jax(images, psf, n_iter=50)
        mean_batch = np.mean(times_batch)
        std_batch = np.std(times_batch)
        print(f"   Average: {mean_batch:.3f}s +/- {std_batch:.3f}s")

        # Check correlation with sequential results
        results_seq_arr = np.array(jnp.stack(results_seq))
        corr = calculate_correlation(results_seq_arr, result_batch)
        print(f"   Correlation with sequential: {corr:.6f}")

        batch_speedup = mean_seq / mean_batch
        print(f"   Speedup vs sequential: {batch_speedup:.2f}x")
        print(f"   Throughput: {batch_size / mean_batch:.2f} images/sec")
    except Exception as e:
        print(f"   FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Image Size':<20} {'C':<12} {'JAX Orig':<12} {'JAX Opt':<12} {'Opt/Orig':<10} {'Opt/C'}")
    print("-"*80)
    for size, data in results.items():
        c_time = f"{data['c']:.3f}s" if data.get('c') else "N/A"
        orig = f"{data['jax_orig']:.3f}s" if data.get('jax_orig') else "N/A"
        opt = f"{data['jax_opt']:.3f}s" if data.get('jax_opt') else "N/A"
        if data.get('jax_orig') and data.get('jax_opt'):
            speedup = f"{data['jax_orig']/data['jax_opt']:.2f}x"
        else:
            speedup = "N/A"
        if data.get('c') and data.get('jax_opt'):
            vs_c = f"{data['jax_opt']/data['c']:.2f}x"
        else:
            vs_c = "N/A"
        print(f"{size:<20} {c_time:<12} {orig:<12} {opt:<12} {speedup:<10} {vs_c}")

    print("\nKey optimizations applied:")
    print("  1. jax.lax.fori_loop: Compiles entire iteration loop into single XLA program")
    print("  2. float32 enforcement: Reduces memory, improves GPU performance")
    print("  3. FFT size optimization: Pads to efficient dimensions (2^n * 3^m * 5^k)")
    print("  4. jax.vmap for batching: Processes multiple images in parallel")


if __name__ == "__main__":
    main()

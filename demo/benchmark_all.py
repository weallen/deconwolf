"""
Complete benchmark of all backends: NumPy, JAX, Numba, FFTW
Run with: python demo/benchmark_all.py
"""
import sys
from pathlib import Path
import time
import numpy as np
import tifffile as tf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.dw_numpy import DeconvolutionConfig, deconvolve as deconvolve_numpy
from python.dw_fast import deconvolve_fast


def benchmark_backend(name, backend_func, im, psf, cfg, n_runs=3):
    """Benchmark a backend with multiple runs"""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")

    times = []
    result = None

    for run in range(n_runs):
        try:
            start = time.perf_counter()
            result = backend_func(im, psf, cfg)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"  Run {run+1}/{n_runs}: {elapsed:.3f}s")
        except Exception as e:
            print(f"  Run {run+1}/{n_runs}: FAILED - {e}")
            return None, None, str(e)

    mean_time = np.mean(times)
    std_time = np.std(times)

    print(f"  Average: {mean_time:.3f}s ± {std_time:.3f}s")
    if result is not None:
        print(f"  Output range: [{result.min():.1f}, {result.max():.1f}]")

    return mean_time, result, None


def main():
    demo_dir = Path(__file__).resolve().parent

    # Load data
    print("Loading test data...")
    im = tf.imread(demo_dir / "dapi_data" / "dapi_001.tif").astype(np.float32)
    psf = tf.imread(demo_dir / "dapi_data" / "PSF_dapi.tif").astype(np.float32)
    im_xyz = np.transpose(im, (2, 1, 0))
    psf_xyz = np.transpose(psf, (2, 1, 0))

    print(f"Image shape (XYZ): {im_xyz.shape}")
    print(f"PSF shape (XYZ): {psf_xyz.shape}")
    print(f"Image size: {im_xyz.nbytes / (1024**2):.1f} MB")

    # Configuration
    cfg = DeconvolutionConfig(
        n_iter=20,
        border_quality=2,
        positivity=True,
        metric="idiv",
        start_condition="flat",
        xycropfactor=0.001,
        use_weights=True,
        offset=5.0,
        alphamax=1.0,
        pad_fast_fft=True,
    )

    print(f"\nConfiguration:")
    print(f"  Iterations: {cfg.n_iter}")
    print(f"  Border quality: {cfg.border_quality}")
    print(f"  FFT padding: {cfg.pad_fast_fft}")
    print(f"  Use weights: {cfg.use_weights}")

    # Create output directory
    output_dir = demo_dir / "outputs" / "dapi_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    outputs = {}

    # 0. C Reference (if available)
    print("\n" + "="*60)
    print("0. Testing C Reference (dw)")
    print("="*60)

    dw_path = Path("/Users/wea/src/deconwolf/build/dw")
    if dw_path.exists():
        import subprocess

        times_c = []
        for run in range(3):
            try:
                start = time.perf_counter()
                result = subprocess.run(
                    [str(dw_path), "--iter", "20",
                     str(demo_dir / "dapi_data" / "dapi_001.tif"),
                     str(demo_dir / "dapi_data" / "PSF_dapi.tif"),
                     "--overwrite", "--noplan", "--float",
                     "-o", str(output_dir / "dw_c_benchmark.tif")],
                    cwd=demo_dir,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                elapsed = time.perf_counter() - start
                times_c.append(elapsed)
                print(f"  Run {run+1}/3: {elapsed:.3f}s")
            except Exception as e:
                print(f"  Run {run+1}/3: FAILED - {e}")
                break

        if times_c:
            mean_time_c = np.mean(times_c)
            std_time_c = np.std(times_c)
            results["C (FFTW)"] = mean_time_c
            print(f"  Average: {mean_time_c:.3f}s ± {std_time_c:.3f}s")

            # Load C result
            try:
                result_c = tf.imread(output_dir / "dw_c_benchmark.tif").astype(np.float32)
                outputs["C (FFTW)"] = result_c
                print(f"  Output range: [{result_c.min():.1f}, {result_c.max():.1f}]")
            except:
                pass
    else:
        print(f"  C executable not found at {dw_path}")
        print("  Skipping C benchmark")

    # 1. NumPy (baseline)
    print("\n" + "="*60)
    print("1. Testing NumPy (baseline)")
    print("="*60)
    time_numpy, result_numpy, error = benchmark_backend(
        "NumPy",
        lambda im, psf, cfg: deconvolve_numpy(im, psf, method="shb", cfg=cfg),
        im_xyz, psf_xyz, cfg,
        n_runs=3
    )
    if time_numpy:
        results["NumPy"] = time_numpy
        outputs["NumPy"] = result_numpy

    # 2. JAX
    print("\n" + "="*60)
    print("2. Testing JAX")
    print("="*60)
    try:
        import jax
        print(f"JAX version: {jax.__version__}")
        print(f"JAX devices: {jax.devices()}")

        time_jax, result_jax, error = benchmark_backend(
            "JAX",
            lambda im, psf, cfg: deconvolve_fast(im, psf, method="shb", backend="jax", cfg=cfg),
            im_xyz, psf_xyz, cfg,
            n_runs=3
        )
        if time_jax:
            results["JAX"] = time_jax
            outputs["JAX"] = result_jax

            # Compare with NumPy and C
            if result_numpy is not None:
                diff = np.abs(result_numpy - result_jax)
                print(f"  vs NumPy: mean_diff={diff.mean():.3f}, max_diff={diff.max():.3f}")

            if "C (FFTW)" in outputs:
                result_c_zyx = outputs["C (FFTW)"]
                result_jax_zyx = np.transpose(result_jax, (2, 1, 0))
                diff_c = np.abs(result_c_zyx - result_jax_zyx)
                rel_err = (diff_c / (result_c_zyx + 1e-6)).mean() * 100
                print(f"  vs C: mean_rel_error={rel_err:.4f}%")
    except ImportError as e:
        print(f"JAX not available: {e}")

    # 3. Numba
    print("\n" + "="*60)
    print("3. Testing Numba")
    print("="*60)
    try:
        import numba
        print(f"Numba version: {numba.__version__}")

        time_numba, result_numba, error = benchmark_backend(
            "Numba",
            lambda im, psf, cfg: deconvolve_fast(im, psf, method="shb", backend="numba", cfg=cfg),
            im_xyz, psf_xyz, cfg,
            n_runs=3
        )
        if time_numba:
            results["Numba"] = time_numba
            outputs["Numba"] = result_numba

            # Compare with NumPy and C
            if result_numpy is not None:
                diff = np.abs(result_numpy - result_numba)
                print(f"  vs NumPy: mean_diff={diff.mean():.3f}, max_diff={diff.max():.3f}")

            if "C (FFTW)" in outputs:
                result_c_zyx = outputs["C (FFTW)"]
                result_numba_zyx = np.transpose(result_numba, (2, 1, 0))
                diff_c = np.abs(result_c_zyx - result_numba_zyx)
                rel_err = (diff_c / (result_c_zyx + 1e-6)).mean() * 100
                print(f"  vs C: mean_rel_error={rel_err:.4f}%")
    except ImportError as e:
        print(f"Numba not available: {e}")
        print("Note: Numba requires Python 3.10-3.13 (not 3.14)")

    # 4. FFTW
    print("\n" + "="*60)
    print("4. Testing FFTW")
    print("="*60)
    try:
        import pyfftw
        print(f"PyFFTW version: {pyfftw.__version__}")

        cfg_fftw = DeconvolutionConfig(**cfg.__dict__)
        cfg_fftw.fftw_threads = 4
        print(f"Using {cfg_fftw.fftw_threads} threads")

        time_fftw, result_fftw, error = benchmark_backend(
            "FFTW",
            lambda im, psf, cfg: deconvolve_fast(im, psf, method="shb", backend="fftw", cfg=cfg),
            im_xyz, psf_xyz, cfg_fftw,
            n_runs=3
        )
        if time_fftw:
            results["FFTW"] = time_fftw
            outputs["FFTW"] = result_fftw

            # Compare with NumPy and C
            if result_numpy is not None:
                diff = np.abs(result_numpy - result_fftw)
                print(f"  vs NumPy: mean_diff={diff.mean():.3f}, max_diff={diff.max():.3f}")

            if "C (FFTW)" in outputs:
                result_c_zyx = outputs["C (FFTW)"]
                result_fftw_zyx = np.transpose(result_fftw, (2, 1, 0))
                diff_c = np.abs(result_c_zyx - result_fftw_zyx)
                rel_err = (diff_c / (result_c_zyx + 1e-6)).mean() * 100
                print(f"  vs C: mean_rel_error={rel_err:.4f}%")
    except ImportError as e:
        print(f"FFTW not available: {e}")

    # Summary table
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    if not results:
        print("No successful benchmarks!")
        return

    baseline = results.get("NumPy")
    if baseline:
        print(f"\n{'Backend':<20} {'Time (s)':<12} {'Speedup':<10} {'Status'}")
        print("-"*80)

        # Sort by time (fastest first)
        for name in sorted(results.keys(), key=lambda k: results[k]):
            time_val = results[name]
            speedup = baseline / time_val if baseline else 1.0

            print(f"{name:<20} {time_val:>6.3f}s      {speedup:>5.2f}x      ✓")

        print("="*80)

        # Find fastest
        fastest_name = min(results.keys(), key=lambda k: results[k])
        fastest_speedup = baseline / results[fastest_name]
        print(f"\n🏆 Fastest: {fastest_name} ({fastest_speedup:.2f}x faster than NumPy)")

    # Save results
    result_file = demo_dir / "outputs" / "benchmarks" / "dapi_dataset_results.txt"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to {result_file}")
    with open(result_file, "w") as f:
        f.write("Deconvolution Backend Benchmark\n")
        f.write("="*80 + "\n\n")
        f.write(f"Image: {im_xyz.shape}\n")
        f.write(f"Iterations: {cfg.n_iter}\n")
        f.write(f"Border quality: {cfg.border_quality}\n\n")

        for name, time_val in sorted(results.items(), key=lambda x: x[1]):
            speedup = baseline / time_val if baseline else 1.0
            f.write(f"{name:<15} {time_val:>6.3f}s   {speedup:>5.2f}x\n")


if __name__ == "__main__":
    main()

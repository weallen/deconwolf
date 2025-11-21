"""
Benchmark all backends on the new test dataset (input.tif, psf.tif)
Run with: python demo/benchmark_new_dataset.py
"""
import sys
from pathlib import Path
import time
import numpy as np
import tifffile as tf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dwpy.dw_numpy import DeconvolutionConfig, deconvolve as deconvolve_numpy
from dwpy.dw_fast import deconvolve_fast


def benchmark_backend(name, backend_func, im, psf, cfg, n_runs=3):
    """Benchmark a backend with multiple runs"""
    print(f"\n{'='*60}")
    print(f"{name}")
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
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    mean_time = np.mean(times)
    std_time = np.std(times)
    best_time = np.min(times)

    print(f"  Average: {mean_time:.3f}s ± {std_time:.3f}s (best: {best_time:.3f}s)")
    if result is not None:
        print(f"  Output: [{result.min():.1f}, {result.max():.1f}]")

    return mean_time, result


def calculate_reconstruction_metrics(output, ground_truth):
    """Calculate reconstruction quality metrics against ground truth"""
    # Ensure same shape
    if output.shape != ground_truth.shape:
        print(f"  Warning: shape mismatch {output.shape} vs {ground_truth.shape}")
        return {}

    # Mean Squared Error
    mse = np.mean((output - ground_truth) ** 2)

    # Peak Signal-to-Noise Ratio
    max_val = ground_truth.max()
    psnr = 10 * np.log10((max_val ** 2) / mse) if mse > 0 else float('inf')

    # Normalized Root Mean Square Error
    nrmse = np.sqrt(mse) / (ground_truth.max() - ground_truth.min())

    # Relative error
    rel_error = np.mean(np.abs(output - ground_truth) / (ground_truth + 1e-6)) * 100

    return {
        "mse": mse,
        "psnr": psnr,
        "nrmse": nrmse,
        "rel_error": rel_error,
    }


def main():
    demo_dir = Path(__file__).resolve().parent

    # Load new dataset from synthetic_data folder
    print("="*80)
    print("Loading synthetic test dataset...")
    print("="*80)

    data_dir = demo_dir / "synthetic_data"
    input_path = data_dir / "input.tif"
    psf_path = data_dir / "psf.tif"
    ground_truth_path = data_dir / "ground-truth.tif"

    if not input_path.exists():
        print(f"ERROR: {input_path} not found!")
        print("Please ensure synthetic_data/input.tif exists")
        return

    if not psf_path.exists():
        print(f"ERROR: {psf_path} not found!")
        print("Please ensure synthetic_data/psf.tif exists")
        return

    im = tf.imread(input_path).astype(np.float32)
    psf = tf.imread(psf_path).astype(np.float32)

    # Load ground truth
    ground_truth = None
    if ground_truth_path.exists():
        ground_truth = tf.imread(ground_truth_path).astype(np.float32)
        print(f"✓ Ground truth loaded: {ground_truth.shape}")
    else:
        print(f"⚠ Ground truth not found at {ground_truth_path}")

    print(f"Input image shape (ZYX): {im.shape}")
    print(f"PSF shape (ZYX): {psf.shape}")
    print(f"Input image size: {im.nbytes / (1024**2):.1f} MB")
    print(f"Input image range: [{im.min():.1f}, {im.max():.1f}]")
    print(f"PSF sum: {psf.sum():.6f}")

    # Convert to XYZ (internal format)
    im_xyz = np.transpose(im, (2, 1, 0))
    psf_xyz = np.transpose(psf, (2, 1, 0))

    print(f"Internal format (XYZ): im={im_xyz.shape}, psf={psf_xyz.shape}")

    # Auto-determine good configuration
    from python.dw_auto import auto_config

    cfg = auto_config(im_xyz, psf_xyz, quality="balanced")

    # Store reconstruction metrics
    reconstruction_metrics = {}

    # Create output directory
    output_dir = demo_dir / "outputs" / "synthetic_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nAuto-selected configuration:")
    print(f"  Iterations: {cfg.n_iter}")
    print(f"  Border quality: {cfg.border_quality}")
    print(f"  Offset: {cfg.offset:.2f}")
    print(f"  Tile size: {cfg.tile_max_size}")
    print(f"  Tile overlap: {cfg.tile_overlap}")
    print(f"  FFT padding: {cfg.pad_fast_fft}")
    print(f"  Use weights: {cfg.use_weights}")

    results = {}
    outputs = {}

    # 1. C Reference (if available)
    print("\n" + "="*80)
    print("BASELINE: C Reference (dw)")
    print("="*80)

    dw_path = Path("/Users/wea/src/deconwolf/build/dw")
    if dw_path.exists():
        import subprocess

        times_c = []
        for run in range(3):
            try:
                start = time.perf_counter()
                result = subprocess.run(
                    [str(dw_path), "--iter", str(cfg.n_iter),
                     str(input_path), str(psf_path),
                     "--overwrite", "--noplan", "--float",
                     "-o", str(output_dir / "output_c.tif")],
                    cwd=demo_dir,
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                elapsed = time.perf_counter() - start
                times_c.append(elapsed)
                print(f"  Run {run+1}/3: {elapsed:.3f}s")
            except Exception as e:
                print(f"  FAILED: {e}")
                break

        if times_c:
            mean_time_c = np.mean(times_c)
            results["C (FFTW)"] = mean_time_c
            print(f"  Average: {mean_time_c:.3f}s ± {np.std(times_c):.3f}s")

            # Load C result
            try:
                result_c = tf.imread(output_dir / "output_c.tif").astype(np.float32)
                outputs["C (FFTW)"] = result_c
                print(f"  Output: [{result_c.min():.1f}, {result_c.max():.1f}]")

                # Calculate reconstruction metrics vs ground truth
                if ground_truth is not None:
                    metrics = calculate_reconstruction_metrics(result_c, ground_truth)
                    reconstruction_metrics["C (FFTW)"] = metrics
                    print(f"  vs Ground Truth: PSNR={metrics['psnr']:.2f}dB, NRMSE={metrics['nrmse']:.4f}, RelErr={metrics['rel_error']:.2f}%")
            except Exception as e:
                print(f"  Failed to load C output: {e}")

    # 2. NumPy
    time_numpy, result_numpy = benchmark_backend(
        "NumPy (baseline)",
        lambda im, psf, cfg: deconvolve_numpy(im, psf, method="shb", cfg=cfg),
        im_xyz, psf_xyz, cfg,
        n_runs=3
    )
    if time_numpy:
        results["NumPy"] = time_numpy
        outputs["NumPy"] = result_numpy

        # Save NumPy output
        out_zyx = np.transpose(result_numpy, (2, 1, 0))
        tf.imwrite(output_dir / "output_numpy.tif", out_zyx)

        # Calculate reconstruction metrics vs ground truth
        if ground_truth is not None:
            metrics = calculate_reconstruction_metrics(out_zyx, ground_truth)
            reconstruction_metrics["NumPy"] = metrics
            print(f"  vs Ground Truth: PSNR={metrics['psnr']:.2f}dB, NRMSE={metrics['nrmse']:.4f}, RelErr={metrics['rel_error']:.2f}%")

    # 3. JAX
    try:
        import jax
        print(f"\nJAX version: {jax.__version__}, devices: {jax.devices()}")

        time_jax, result_jax = benchmark_backend(
            "JAX",
            lambda im, psf, cfg: deconvolve_fast(im, psf, method="shb", backend="jax", cfg=cfg),
            im_xyz, psf_xyz, cfg,
            n_runs=3
        )
        if time_jax:
            results["JAX"] = time_jax
            outputs["JAX"] = result_jax

            # Save JAX output
            out_zyx = np.transpose(result_jax, (2, 1, 0))
            tf.imwrite(output_dir / "output_jax.tif", out_zyx)

            # Compare
            if result_numpy is not None:
                diff = np.abs(result_numpy - result_jax)
                print(f"  vs NumPy: mean_diff={diff.mean():.3f}, max_diff={diff.max():.3f}")

            # Calculate reconstruction metrics vs ground truth
            if ground_truth is not None:
                result_jax_zyx = np.transpose(result_jax, (2, 1, 0))
                metrics = calculate_reconstruction_metrics(result_jax_zyx, ground_truth)
                reconstruction_metrics["JAX"] = metrics
                print(f"  vs Ground Truth: PSNR={metrics['psnr']:.2f}dB, NRMSE={metrics['nrmse']:.4f}, RelErr={metrics['rel_error']:.2f}%")
    except ImportError:
        print("\nJAX not available")

    # 4. Numba
    try:
        import numba
        print(f"\nNumba version: {numba.__version__}")

        time_numba, result_numba = benchmark_backend(
            "Numba",
            lambda im, psf, cfg: deconvolve_fast(im, psf, method="shb", backend="numba", cfg=cfg),
            im_xyz, psf_xyz, cfg,
            n_runs=3
        )
        if time_numba:
            results["Numba"] = time_numba
            outputs["Numba"] = result_numba

            # Save Numba output
            out_zyx = np.transpose(result_numba, (2, 1, 0))
            tf.imwrite(output_dir / "output_numba.tif", out_zyx)

            # Compare
            if result_numpy is not None:
                diff = np.abs(result_numpy - result_numba)
                print(f"  vs NumPy: mean_diff={diff.mean():.3f}, max_diff={diff.max():.3f}")

            # Calculate reconstruction metrics vs ground truth
            if ground_truth is not None:
                result_numba_zyx = np.transpose(result_numba, (2, 1, 0))
                metrics = calculate_reconstruction_metrics(result_numba_zyx, ground_truth)
                reconstruction_metrics["Numba"] = metrics
                print(f"  vs Ground Truth: PSNR={metrics['psnr']:.2f}dB, NRMSE={metrics['nrmse']:.4f}, RelErr={metrics['rel_error']:.2f}%")
    except ImportError:
        print("\nNumba not available (requires Python 3.10-3.13)")

    # 5. FFTW
    try:
        import pyfftw
        print(f"\nPyFFTW version: {pyfftw.__version__}")

        cfg_fftw = DeconvolutionConfig(**cfg.__dict__)
        cfg_fftw.fftw_threads = 4

        time_fftw, result_fftw = benchmark_backend(
            "FFTW (4 threads)",
            lambda im, psf, cfg: deconvolve_fast(im, psf, method="shb", backend="fftw", cfg=cfg),
            im_xyz, psf_xyz, cfg_fftw,
            n_runs=3
        )
        if time_fftw:
            results["FFTW"] = time_fftw
            outputs["FFTW"] = result_fftw

            # Save FFTW output
            out_zyx = np.transpose(result_fftw, (2, 1, 0))
            tf.imwrite(output_dir / "output_fftw.tif", out_zyx)

            # Compare
            if result_numpy is not None:
                diff = np.abs(result_numpy - result_fftw)
                print(f"  vs NumPy: mean_diff={diff.mean():.3f}, max_diff={diff.max():.3f}")

            # Calculate reconstruction metrics vs ground truth
            if ground_truth is not None:
                result_fftw_zyx = np.transpose(result_fftw, (2, 1, 0))
                metrics = calculate_reconstruction_metrics(result_fftw_zyx, ground_truth)
                reconstruction_metrics["FFTW"] = metrics
                print(f"  vs Ground Truth: PSNR={metrics['psnr']:.2f}dB, NRMSE={metrics['nrmse']:.4f}, RelErr={metrics['rel_error']:.2f}%")
    except ImportError:
        print("\nPyFFTW not available")

    # Summary table
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    if not results:
        print("No successful benchmarks!")
        return

    baseline = results.get("NumPy")
    print(f"\n{'Backend':<20} {'Time (s)':<12} {'Speedup':<12} {'vs NumPy'}")
    print("-"*80)

    # Sort by time (fastest first)
    for name in sorted(results.keys(), key=lambda k: results[k]):
        time_val = results[name]
        speedup = baseline / time_val if baseline else 1.0
        marker = "🏆" if name == min(results.keys(), key=lambda k: results[k]) else "  "

        print(f"{marker} {name:<18} {time_val:>6.3f}s      {speedup:>6.2f}x       ✓")

    print("="*80)

    # Reconstruction quality table
    if reconstruction_metrics:
        print("\n" + "="*80)
        print("RECONSTRUCTION QUALITY (vs Ground Truth)")
        print("="*80)
        print(f"\n{'Backend':<20} {'PSNR (dB)':<12} {'NRMSE':<10} {'RelErr (%)'}")
        print("-"*80)

        # Sort by PSNR (higher is better)
        for name in sorted(reconstruction_metrics.keys(), key=lambda k: reconstruction_metrics[k]['psnr'], reverse=True):
            metrics = reconstruction_metrics[name]
            marker = "🏆" if name == max(reconstruction_metrics.keys(), key=lambda k: reconstruction_metrics[k]['psnr']) else "  "
            print(f"{marker} {name:<18} {metrics['psnr']:>8.2f}      {metrics['nrmse']:>6.4f}     {metrics['rel_error']:>6.2f}")

        print("="*80)
        best_quality = max(reconstruction_metrics.keys(), key=lambda k: reconstruction_metrics[k]['psnr'])
        print(f"🏆 Best reconstruction quality: {best_quality} (PSNR={reconstruction_metrics[best_quality]['psnr']:.2f}dB)")

    # Additional insights
    if "C (FFTW)" in results and baseline:
        c_speedup = results["C (FFTW)"] / baseline
        print(f"\n📊 Python NumPy is {1/c_speedup:.2f}x faster than C!")

    fastest = min(results.keys(), key=lambda k: results[k])
    if baseline:
        print(f"⚡ Fastest overall: {fastest} ({baseline/results[fastest]:.2f}x faster than NumPy)")

    print(f"\n📁 Outputs saved to {output_dir}:")
    for name in outputs.keys():
        backend_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        print(f"  - output_{backend_name}.tif")

    # Save benchmark results
    result_file = demo_dir / "outputs" / "benchmarks" / "synthetic_dataset_results.txt"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, "w") as f:
        f.write("Synthetic Dataset Benchmark Results\n")
        f.write("="*80 + "\n\n")
        f.write(f"Image: {im_xyz.shape} XYZ\n")
        f.write(f"Iterations: {cfg.n_iter}\n\n")
        f.write("SPEED:\n")
        for name, time_val in sorted(results.items(), key=lambda x: x[1]):
            speedup = baseline / time_val if baseline else 1.0
            f.write(f"  {name:<20} {time_val:>6.3f}s   {speedup:>5.2f}x\n")
        if reconstruction_metrics:
            f.write("\nRECONSTRUCTION QUALITY (vs Ground Truth):\n")
            for name in sorted(reconstruction_metrics.keys(), key=lambda k: reconstruction_metrics[k]['psnr'], reverse=True):
                m = reconstruction_metrics[name]
                f.write(f"  {name:<20} PSNR={m['psnr']:>6.2f}dB  NRMSE={m['nrmse']:.4f}  RelErr={m['rel_error']:.2f}%\n")
    print(f"Benchmark summary saved to: {result_file}")


if __name__ == "__main__":
    main()

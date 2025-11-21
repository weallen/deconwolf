"""
Comprehensive benchmark on synthetic dataset with ground truth.
Tests all available backends (NumPy, JAX, CuPy, Numba, FFTW) with:
  - C-matching parameters (50 iterations)
  - Auto-tuned parameters (50 iterations)

Outputs quality metrics (PSNR, NRMSE) and performance comparisons.
"""
import sys
from pathlib import Path
import time
from contextlib import contextmanager, nullcontext
import numpy as np
import tifffile as tf
import subprocess

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dwpy.dw_numpy import DeconvolutionConfig, deconvolve as deconvolve_numpy
from dwpy.dw_fast import deconvolve_fast


def center_psf_for_c(psf):
    """
    Center PSF for C code compatibility.
    C expects PSF max at (0,0,0) which will be circshifted during processing.
    """
    # Find maximum location
    max_idx = np.unravel_index(np.argmax(psf), psf.shape)
    print(f"  Original PSF max at: {max_idx}")

    # C code expects PSF with max at (0,0,0)
    # Use numpy roll to shift max to origin
    centered = np.roll(psf, -max_idx[0], axis=0)
    centered = np.roll(centered, -max_idx[1], axis=1)
    centered = np.roll(centered, -max_idx[2], axis=2)

    # Verify
    new_max = np.unravel_index(np.argmax(centered), centered.shape)
    print(f"  Centered PSF max at: {new_max}")
    print(f"  PSF sum: {centered.sum():.6f}")

    # Normalize to sum=1
    centered = centered / centered.sum()
    print(f"  After normalization: sum={centered.sum():.6f}")

    return centered


def calculate_metrics(output, ground_truth, name=""):
    """Calculate and display reconstruction quality metrics"""
    if output.shape != ground_truth.shape:
        print(f"  ⚠ Shape mismatch: {output.shape} vs {ground_truth.shape}")
        return None

    mse = np.mean((output - ground_truth) ** 2)
    max_val = ground_truth.max()
    psnr = 10 * np.log10((max_val ** 2) / mse) if mse > 0 else float('inf')
    nrmse = np.sqrt(mse) / (ground_truth.max() - ground_truth.min())
    rel_error = np.mean(np.abs(output - ground_truth) / (ground_truth + 1e-6)) * 100

    print(f"  {name}")
    print(f"    PSNR: {psnr:.2f} dB")
    print(f"    NRMSE: {nrmse:.4f}")
    print(f"    RelErr: {rel_error:.2f}%")
    print(f"    Range: [{output.min():.1f}, {output.max():.1f}]")

    return {"psnr": psnr, "nrmse": nrmse, "rel_error": rel_error, "mse": mse}


@contextmanager
def log_numpy_iterations():
    """
    Temporarily wrap dw_numpy.iter_shb_step to print per-iteration stats
    (error plus input/output ranges) for debugging numerical issues.
    """
    import dwpy.dw_numpy as dw_numpy

    orig_iter = dw_numpy.iter_shb_step
    counter = {"i": 0}

    def _summary(arr):
        return (
            float(np.nanmin(arr)),
            float(np.nanmax(arr)),
            float(np.nanmean(arr)),
        )

    def wrapped_iter(im, cK, pK, W, bg, metric):
        i = counter["i"]
        counter["i"] += 1
        p_min, p_max, p_mean = _summary(pK)
        xp_new, err = orig_iter(im, cK, pK, W, bg, metric)
        x_min, x_max, x_mean = _summary(xp_new)
        metric_label = "idiv" if metric == "idiv" else f"err[{metric}]"
        print(
            f"iter {i+1:03d}: {metric_label}={err:.3e} err={err:.3e} "
            f"p[min,max,mean]={p_min:.3e},{p_max:.3e},{p_mean:.3e} "
            f"x[min,max,mean]={x_min:.3e},{x_max:.3e},{x_mean:.3e}",
            flush=True,
        )
        return xp_new, err

    dw_numpy.iter_shb_step = wrapped_iter
    try:
        yield
    finally:
        dw_numpy.iter_shb_step = orig_iter


def run_test(name, backend_func, im, psf, cfg, ground_truth, log_iterations=False):
    """Run single test and return metrics"""
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    print(f"Running {cfg.n_iter} iterations...", flush=True)

    start = time.perf_counter()
    log_ctx = log_numpy_iterations() if log_iterations else nullcontext()
    with log_ctx:
        result = backend_func(im, psf, cfg)
    elapsed = time.perf_counter() - start

    print(f"✓ Completed in {elapsed:.2f}s ({elapsed/cfg.n_iter:.2f}s per iteration)")

    # Convert to ZYX for comparison
    result_zyx = np.transpose(result, (2, 1, 0))

    # Calculate metrics
    metrics = calculate_metrics(result_zyx, ground_truth, "Quality vs Ground Truth:")

    return elapsed, result, metrics


def main():
    demo_dir = Path(__file__).resolve().parent
    data_dir = demo_dir / "synthetic_data"
    output_dir = demo_dir / "outputs" / "synthetic_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("="*70)
    print("Loading synthetic dataset with ground truth...")
    print("="*70)

    im = tf.imread(data_dir / "input.tif").astype(np.float32)
    psf_orig = tf.imread(data_dir / "psf.tif").astype(np.float32)
    ground_truth = tf.imread(data_dir / "ground-truth.tif").astype(np.float32)

    print("\nCentering PSF for C compatibility...")
    psf = center_psf_for_c(psf_orig)

    # Save centered PSF for C to use
    tf.imwrite(data_dir / "psf_centered.tif", psf)

    im_xyz = np.transpose(im, (2, 1, 0))
    psf_xyz = np.transpose(psf, (2, 1, 0))

    print(f"Image: {im_xyz.shape} (XYZ)")
    print(f"PSF: {psf_xyz.shape}")
    print(f"Ground truth: {ground_truth.shape} (ZYX)")
    print(f"\n📁 All outputs will be saved to: {output_dir}/")

    # Configuration 1: C-matching parameters (but offset=0 for this dataset)
    print("\n" + "="*70)
    print("CONFIGURATION 1: C-Matching Parameters")
    print("="*70)
    print("Note: Using offset=0 instead of C default offset=5")
    print("      (This dataset has image.min()=0, offset=5 causes overflow)")

    cfg_c_match = DeconvolutionConfig(
        n_iter=50,
        border_quality=2,
        positivity=True,
        metric="idiv",
        start_condition="flat",
        xycropfactor=0.001,
        use_weights=True,
        offset=0.0,  # Use 0 for this dataset (image min=0, not 1758 like DAPI)
        alphamax=1.0,
        pad_fast_fft=False,  # Match C (no padding)
    )

    print(f"Parameters: n_iter={cfg_c_match.n_iter}, border_quality={cfg_c_match.border_quality}")
    print(f"            offset={cfg_c_match.offset}, pad_fast_fft={cfg_c_match.pad_fast_fft}")

    results = {}

    # Test 0: C Reference
    print("\n" + "="*70)
    print("BASELINE: C Reference")
    print("="*70)

    dw_path = Path("/Users/wea/src/deconwolf/build/dw")
    if dw_path.exists():
        print(f"Running C with 50 iterations SHB (offset=0)...", flush=True)
        times_c = []
        for run in range(3):
            try:
                start = time.perf_counter()
                result = subprocess.run(
                    [str(dw_path), "--iter", "50",
                     str(data_dir / "input.tif"),
                     str(data_dir / "psf_centered.tif"),
                     "--overwrite", "--noplan", "--float",
                     "--offset", "0",
                     "--method", "shb",  # Use SHB method
                     "-o", str(output_dir / "output_c_50iter.tif")],
                    cwd=demo_dir,
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                elapsed = time.perf_counter() - start
                times_c.append(elapsed)
                print(f"  Run {run+1}/3: {elapsed:.2f}s")
            except Exception as e:
                print(f"  Run {run+1}/3: FAILED - {e}")
                break

        if times_c:
            mean_time_c = np.mean(times_c)
            print(f"✓ Average: {mean_time_c:.2f}s ± {np.std(times_c):.2f}s")

            # Load and evaluate C result
            try:
                result_c = tf.imread(output_dir / "output_c_50iter.tif").astype(np.float32)
                metrics_c = calculate_metrics(result_c, ground_truth, "Quality vs Ground Truth:")
                results["C (FFTW)"] = {"time": mean_time_c, "metrics": metrics_c}
            except Exception as e:
                print(f"  Failed to load C output: {e}")
    else:
        print(f"  C executable not found at {dw_path}")
        print("  Skipping C benchmark")

    # Test 1: NumPy with C-matching params
    t1, r1, m1 = run_test(
        "NumPy (SHB, C-matching params)",
        lambda im, psf, cfg: deconvolve_numpy(im, psf, method="shb", cfg=cfg),
        im_xyz, psf_xyz, cfg_c_match, ground_truth,
        log_iterations=True,
    )
    results["NumPy (C-match)"] = {"time": t1, "metrics": m1}

    # Save output
    out_zyx = np.transpose(r1, (2, 1, 0))
    tf.imwrite(output_dir / "numpy_c_match_50iter.tif", out_zyx)

    # Test 2: JAX with C-matching params
    try:
        import jax
        t2, r2, m2 = run_test(
            "JAX (SHB, C-matching params)",
            lambda im, psf, cfg: deconvolve_fast(im, psf, method="shb", backend="jax", cfg=cfg),
            im_xyz, psf_xyz, cfg_c_match, ground_truth
        )
        results["JAX (C-match)"] = {"time": t2, "metrics": m2}

        # Save output
        out_zyx = np.transpose(r2, (2, 1, 0))
        tf.imwrite(output_dir / "jax_c_match_50iter.tif", out_zyx)
    except ImportError:
        print("\n⚠ JAX not available, skipping JAX test")

    # Test 3: CuPy with C-matching params
    try:
        import cupy
        t3, r3, m3 = run_test(
            "CuPy (SHB, C-matching params)",
            lambda im, psf, cfg: deconvolve_fast(im, psf, method="shb", backend="cupy", cfg=cfg),
            im_xyz, psf_xyz, cfg_c_match, ground_truth
        )
        results["CuPy (C-match)"] = {"time": t3, "metrics": m3}

        # Save output
        out_zyx = np.transpose(r3, (2, 1, 0))
        tf.imwrite(output_dir / "cupy_c_match_50iter.tif", out_zyx)
    except ImportError:
        print("\n⚠ CuPy not available, skipping CuPy test")

    # Test 4: Numba with C-matching params
    try:
        import numba
        t4, r4, m4 = run_test(
            "Numba (SHB, C-matching params)",
            lambda im, psf, cfg: deconvolve_fast(im, psf, method="shb", backend="numba", cfg=cfg),
            im_xyz, psf_xyz, cfg_c_match, ground_truth
        )
        results["Numba (C-match)"] = {"time": t4, "metrics": m4}

        # Save output
        out_zyx = np.transpose(r4, (2, 1, 0))
        tf.imwrite(output_dir / "numba_c_match_50iter.tif", out_zyx)
    except ImportError:
        print("\n⚠ Numba not available, skipping Numba test")

    # Test 5: FFTW with C-matching params
    try:
        import pyfftw
        t5, r5, m5 = run_test(
            "FFTW (SHB, C-matching params)",
            lambda im, psf, cfg: deconvolve_fast(im, psf, method="shb", backend="fftw", cfg=cfg),
            im_xyz, psf_xyz, cfg_c_match, ground_truth
        )
        results["FFTW (C-match)"] = {"time": t5, "metrics": m5}

        # Save output
        out_zyx = np.transpose(r5, (2, 1, 0))
        tf.imwrite(output_dir / "fftw_c_match_50iter.tif", out_zyx)
    except ImportError:
        print("\n⚠ PyFFTW not available, skipping FFTW test")

    # Configuration 2: Auto-tuned parameters
    print("\n" + "="*70)
    print("CONFIGURATION 2: Auto-Tuned Parameters")
    print("="*70)

    from dwpy.dw_auto import auto_config

    cfg_auto = auto_config(im_xyz, psf_xyz, quality="high")
    cfg_auto.n_iter = 50  # Override to match test

    print(f"Parameters: n_iter={cfg_auto.n_iter}, border_quality={cfg_auto.border_quality}")
    print(f"            offset={cfg_auto.offset:.2f}, pad_fast_fft={cfg_auto.pad_fast_fft}")
    print(f"            tile_max_size={cfg_auto.tile_max_size}")

    # Test 6: NumPy with auto params
    t6, r6, m6 = run_test(
        "NumPy (SHB, auto-tuned)",
        lambda im, psf, cfg: deconvolve_numpy(im, psf, method="shb", cfg=cfg),
        im_xyz, psf_xyz, cfg_auto, ground_truth,
        log_iterations=True,
    )
    results["NumPy (auto)"] = {"time": t6, "metrics": m6}

    # Save output
    out_zyx = np.transpose(r6, (2, 1, 0))
    tf.imwrite(output_dir / "numpy_auto_50iter.tif", out_zyx)

    # Test 7: JAX with auto params
    try:
        import jax
        t7, r7, m7 = run_test(
            "JAX (SHB, auto-tuned)",
            lambda im, psf, cfg: deconvolve_fast(im, psf, method="shb", backend="jax", cfg=cfg),
            im_xyz, psf_xyz, cfg_auto, ground_truth
        )
        results["JAX (auto)"] = {"time": t7, "metrics": m7}

        # Save output
        out_zyx = np.transpose(r7, (2, 1, 0))
        tf.imwrite(output_dir / "jax_auto_50iter.tif", out_zyx)
    except ImportError:
        print("\n⚠ JAX not available for auto-tuned test")

    # Test 8: CuPy with auto params
    try:
        import cupy
        t8, r8, m8 = run_test(
            "CuPy (SHB, auto-tuned)",
            lambda im, psf, cfg: deconvolve_fast(im, psf, method="shb", backend="cupy", cfg=cfg),
            im_xyz, psf_xyz, cfg_auto, ground_truth
        )
        results["CuPy (auto)"] = {"time": t8, "metrics": m8}

        # Save output
        out_zyx = np.transpose(r8, (2, 1, 0))
        tf.imwrite(output_dir / "cupy_auto_50iter.tif", out_zyx)
    except ImportError:
        print("\n⚠ CuPy not available for auto-tuned test")

    # Test 9: Numba with auto params
    try:
        import numba
        t9, r9, m9 = run_test(
            "Numba (SHB, auto-tuned)",
            lambda im, psf, cfg: deconvolve_fast(im, psf, method="shb", backend="numba", cfg=cfg),
            im_xyz, psf_xyz, cfg_auto, ground_truth
        )
        results["Numba (auto)"] = {"time": t9, "metrics": m9}

        # Save output
        out_zyx = np.transpose(r9, (2, 1, 0))
        tf.imwrite(output_dir / "numba_auto_50iter.tif", out_zyx)
    except ImportError:
        print("\n⚠ Numba not available for auto-tuned test")

    # Test 10: FFTW with auto params
    try:
        import pyfftw
        t10, r10, m10 = run_test(
            "FFTW (SHB, auto-tuned)",
            lambda im, psf, cfg: deconvolve_fast(im, psf, method="shb", backend="fftw", cfg=cfg),
            im_xyz, psf_xyz, cfg_auto, ground_truth
        )
        results["FFTW (auto)"] = {"time": t10, "metrics": m10}

        # Save output
        out_zyx = np.transpose(r10, (2, 1, 0))
        tf.imwrite(output_dir / "fftw_auto_50iter.tif", out_zyx)
    except ImportError:
        print("\n⚠ PyFFTW not available for auto-tuned test")

    # Summary tables
    print("\n" + "="*70)
    print("SUMMARY: RECONSTRUCTION QUALITY (50 iterations)")
    print("="*70)
    print(f"\n{'Configuration':<28} {'PSNR (dB)':<12} {'NRMSE':<10} {'Time (s)'}")
    print("-"*70)

    for name, data in sorted(results.items(), key=lambda x: x[1]['metrics']['psnr'], reverse=True):
        m = data['metrics']
        t = data['time']
        marker = "🏆" if m['psnr'] == max(r['metrics']['psnr'] for r in results.values()) else "  "
        speed_marker = "⚡" if t == min(r['time'] for r in results.values()) else "  "
        print(f"{marker}{speed_marker} {name:<25} {m['psnr']:>8.2f}      {m['nrmse']:>6.4f}     {t:>6.1f}")

    print("="*70)

    # Find best quality
    best = max(results.items(), key=lambda x: x[1]['metrics']['psnr'])
    print(f"\n🏆 Best quality: {best[0]} (PSNR={best[1]['metrics']['psnr']:.2f} dB)")

    # Find fastest
    fastest = min(results.items(), key=lambda x: x[1]['time'])
    print(f"⚡ Fastest: {fastest[0]} ({fastest[1]['time']:.1f}s)")

    # Compare vs C if available
    if "C (FFTW)" in results:
        c_time = results["C (FFTW)"]["time"]
        for name, data in results.items():
            if name != "C (FFTW)":
                speedup = c_time / data["time"]
                if speedup > 1:
                    print(f"   {name} is {speedup:.2f}x faster than C")

    # Compare C-match vs Auto
    if "NumPy (C-match)" in results and "NumPy (auto)" in results:
        c_psnr = results["NumPy (C-match)"]["metrics"]["psnr"]
        auto_psnr = results["NumPy (auto)"]["metrics"]["psnr"]
        improvement = auto_psnr - c_psnr
        print(f"\n📊 Auto-tuning vs C-matching: {improvement:+.2f} dB PSNR improvement")

    # Save summary
    result_file = demo_dir / "outputs" / "benchmarks" / "synthetic_50iter_results.txt"
    with open(result_file, "w") as f:
        f.write("Synthetic Dataset - 50 Iteration Quality Comparison\n")
        f.write("="*70 + "\n\n")
        for name, data in sorted(results.items(), key=lambda x: x[1]['metrics']['psnr'], reverse=True):
            m = data['metrics']
            t = data['time']
            f.write(f"{name:<25} PSNR={m['psnr']:>6.2f}dB  NRMSE={m['nrmse']:.4f}  Time={t:>6.1f}s\n")

    print(f"\nResults saved to: {result_file}")
    print(f"Outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()

"""
Run JAX SHB deconvolution on DAPI dataset with three-way PSF comparison.

This script:
  - Loads DAPI dataset
  - Uses C benchmark result (dw_c_benchmark.tif) as ground truth if available
  - Tests THREE PSF models:
    1. Supplied PSF (PSF_dapi.tif if available, 181×181×79)
    2. Gibson-Lanni PSF (generated, compact: ~17×17×21)
    3. Born-Wolf PSF (generated, compact: ~17×17×21)
  - Runs JAX SHB deconvolution with 50 iterations for each
  - Compares against C benchmark (PSNR, NRMSE, RelErr)
  - Demonstrates memory-efficient compact PSF sizing (99.77% smaller!)

Usage (from repo root):
    python demo/run_dapi_jax_shb.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import time

import numpy as np
import tifffile as tf

# Ensure repo root on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dwpy


def calculate_metrics(output, ground_truth, name=""):
    """Calculate reconstruction quality metrics"""
    if output.shape != ground_truth.shape:
        print(f"  ⚠ Shape mismatch: {output.shape} vs {ground_truth.shape}")
        return None

    mse = np.mean((output - ground_truth) ** 2)
    max_val = ground_truth.max()
    psnr = 10 * np.log10((max_val ** 2) / mse) if mse > 0 else float('inf')
    nrmse = np.sqrt(mse) / (ground_truth.max() - ground_truth.min())
    rel_error = np.mean(np.abs(output - ground_truth) / (ground_truth + 1e-6)) * 100

    print(f"{name}")
    print(f"  PSNR:   {psnr:.2f} dB")
    print(f"  NRMSE:  {nrmse:.4f}")
    print(f"  RelErr: {rel_error:.2f}%")
    print(f"  Range:  [{output.min():.1f}, {output.max():.1f}]")

    return {"psnr": psnr, "nrmse": nrmse, "rel_error": rel_error, "mse": mse}


def main():
    # Setup paths
    demo_dir = Path(__file__).resolve().parent
    data_dir = demo_dir / "dapi_data"
    output_dir = demo_dir / "outputs" / "dapi_jax_shb"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("JAX SHB DECONVOLUTION - DAPI DATASET")
    print("=" * 70)

    # Load data
    print("\nLoading DAPI dataset...")
    im_zyx = tf.imread(data_dir / "dapi_001.tif").astype(np.float32)

    # Load C benchmark as ground truth reference
    ground_truth_path = demo_dir / "outputs" / "dapi_dataset" / "dw_c_benchmark.tif"
    ground_truth = None
    if ground_truth_path.exists():
        ground_truth = tf.imread(ground_truth_path).astype(np.float32)
        print(f"  Ground truth (C benchmark): {ground_truth.shape} (ZYX)")
    else:
        print(f"  Ground truth not found at {ground_truth_path}")

    # Convert to XYZ for processing
    im_xyz = np.transpose(im_zyx, (2, 1, 0))
    M, N, P = im_xyz.shape

    print(f"  Image: {im_xyz.shape} (XYZ) = {im_zyx.shape} (ZYX)")

    # Check if supplied PSF exists
    psf_supplied = None
    if (data_dir / "PSF_dapi.tif").exists():
        psf_supplied_zyx = tf.imread(data_dir / "PSF_dapi.tif").astype(np.float32)
        psf_supplied = np.transpose(psf_supplied_zyx, (2, 1, 0))
        psf_supplied = psf_supplied / psf_supplied.sum()
        print(f"  Supplied PSF: {psf_supplied.shape} (XYZ)")

    # Load experiment configuration from YAML
    print("\n" + "=" * 70)
    print("LOADING CONFIGURATION")
    print("=" * 70)

    config_path = ROOT / "configs" / "dapi_60x_oil.yaml"
    config = dwpy.load_experiment_config(config_path)

    print(f"✓ Config: {config.name}")
    print(f"  Microscope: {config.microscope.objective}")
    print(f"  Voxel size: {config.imaging.dxy*1000:.0f}nm/{config.imaging.dz*1000:.0f}nm")
    print(f"  Wavelength: {config.imaging.wavelength*1000:.0f}nm")
    print(f"  NA={config.microscope.NA}, ni={config.microscope.ni}, ns={config.microscope.ns}")

    # Generate PSFs using configuration
    print("\n" + "=" * 70)
    print("GENERATING PSFs FROM CONFIG (AUTO-SIZED)")
    print("=" * 70)

    # Calculate auto size
    xy_size, z_size = dwpy.auto_psf_size_c_heuristic(
        dxy=config.imaging.dxy,
        dz=config.imaging.dz,
        NA=config.microscope.NA,
        wvl=config.imaging.wavelength,
        ni=config.microscope.ni
    )
    print(f"Auto-calculated PSF size: {xy_size}×{xy_size}×{z_size}")

    # 1. Gibson-Lanni PSF from config
    print("\n1. Gibson-Lanni PSF (from config)")
    psf_gl = dwpy.generate_psf_from_config(config)
    print(f"   Generated: {psf_gl.shape}, sum={psf_gl.sum():.6f}")
    tf.imwrite(output_dir / "PSF_GL.tif", np.transpose(psf_gl, (2, 1, 0)))

    # 2. Born-Wolf PSF (modify config to use BW)
    print("\n2. Born-Wolf PSF (config → BW model)")
    config_bw = dwpy.ExperimentConfig.from_dict(config.to_dict())
    config_bw.psf.model = "bw"
    psf_bw = dwpy.generate_psf_from_config(config_bw)
    print(f"   Generated: {psf_bw.shape}, sum={psf_bw.sum():.6f}")
    tf.imwrite(output_dir / "PSF_BW.tif", np.transpose(psf_bw, (2, 1, 0)))

    # Create DeconvolutionConfig from config file
    cfg = dwpy.DeconvolutionConfig(
        n_iter=config.deconvolution.n_iter,
        border_quality=config.deconvolution.border_quality,
        positivity=config.deconvolution.positivity,
        metric=config.deconvolution.metric,
        use_weights=config.deconvolution.use_weights,
        offset=config.deconvolution.offset,
        pad_fast_fft=config.deconvolution.pad_fast_fft,
        alphamax=config.deconvolution.alphamax,
    )

    print(f"\nDeconvolution config from file:")
    print(f"  {cfg.n_iter} iterations, {config.deconvolution.method} method")
    print(f"  Backend: {config.deconvolution.backend}")
    print(f"  offset={cfg.offset}")

    results = {}

    # Run with supplied PSF (if available)
    if psf_supplied is not None:
        print("\n" + "=" * 70)
        print("DECONVOLUTION WITH SUPPLIED PSF")
        print("=" * 70)

        try:
            import jax
            print(f"JAX backend: {jax.default_backend()}")

            start = time.perf_counter()
            result_supplied = dwpy.deconvolve_fast(
                im_xyz, psf_supplied,
                method="shb",
                backend="jax",
                cfg=cfg
            )
            elapsed_supplied = time.perf_counter() - start

            print(f"✓ Completed in {elapsed_supplied:.2f}s ({elapsed_supplied/cfg.n_iter:.2f}s/iter)")

            # Save output
            result_supplied_zyx = np.transpose(result_supplied, (2, 1, 0))
            tf.imwrite(output_dir / "output_JAX_SHB_Supplied.tif", result_supplied_zyx)
            print(f"  Saved: output_JAX_SHB_Supplied.tif")

            # Calculate metrics if ground truth available
            metrics_supplied = None
            if ground_truth is not None:
                metrics_supplied = calculate_metrics(result_supplied_zyx, ground_truth, "\nQuality Metrics (Supplied PSF):")
                results["Supplied"] = {"time": elapsed_supplied, "metrics": metrics_supplied}
            else:
                print(f"  Output range: [{result_supplied_zyx.min():.1f}, {result_supplied_zyx.max():.1f}]")
                results["Supplied"] = {"time": elapsed_supplied}

        except ImportError as e:
            print(f"✗ JAX not available: {e}")
            return 1

    # Run with Gibson-Lanni PSF
    print("\n" + "=" * 70)
    print("DECONVOLUTION WITH GIBSON-LANNI PSF")
    print("=" * 70)

    try:
        import jax
        if "Supplied" not in results:
            print(f"JAX backend: {jax.default_backend()}")

        start = time.perf_counter()
        result_gl = dwpy.deconvolve_fast(
            im_xyz, psf_gl,
            method="shb",
            backend="jax",
            cfg=cfg
        )
        elapsed_gl = time.perf_counter() - start

        print(f"✓ Completed in {elapsed_gl:.2f}s ({elapsed_gl/cfg.n_iter:.2f}s/iter)")

        # Save output
        result_gl_zyx = np.transpose(result_gl, (2, 1, 0))
        tf.imwrite(output_dir / "output_JAX_SHB_GL.tif", result_gl_zyx)
        print(f"  Saved: output_JAX_SHB_GL.tif")

        # Calculate metrics if ground truth available
        if ground_truth is not None:
            metrics_gl = calculate_metrics(result_gl_zyx, ground_truth, "\nQuality Metrics (GL PSF):")
            results["GL"] = {"time": elapsed_gl, "metrics": metrics_gl}
        else:
            print(f"  Output range: [{result_gl_zyx.min():.1f}, {result_gl_zyx.max():.1f}]")
            results["GL"] = {"time": elapsed_gl}

    except ImportError as e:
        print(f"✗ JAX not available: {e}")
        return 1

    # Run with Born-Wolf PSF
    print("\n" + "=" * 70)
    print("DECONVOLUTION WITH BORN-WOLF PSF")
    print("=" * 70)

    start = time.perf_counter()
    result_bw = dwpy.deconvolve_fast(
        im_xyz, psf_bw,
        method="shb",
        backend="jax",
        cfg=cfg
    )
    elapsed_bw = time.perf_counter() - start

    print(f"✓ Completed in {elapsed_bw:.2f}s ({elapsed_bw/cfg.n_iter:.2f}s/iter)")

    # Save output
    result_bw_zyx = np.transpose(result_bw, (2, 1, 0))
    tf.imwrite(output_dir / "output_JAX_SHB_BW.tif", result_bw_zyx)
    print(f"  Saved: output_JAX_SHB_BW.tif")

    # Calculate metrics if ground truth available
    if ground_truth is not None:
        metrics_bw = calculate_metrics(result_bw_zyx, ground_truth, "\nQuality Metrics (BW PSF):")
        results["BW"] = {"time": elapsed_bw, "metrics": metrics_bw}
    else:
        print(f"  Output range: [{result_bw_zyx.min():.1f}, {result_bw_zyx.max():.1f}]")
        results["BW"] = {"time": elapsed_bw}

    # Comparison summary
    print("\n" + "=" * 70)
    if ground_truth is not None:
        print("THREE-WAY PSF COMPARISON SUMMARY (vs C Benchmark)")
    else:
        print("PSF COMPARISON SUMMARY")
    print("=" * 70)

    # If we have metrics, show quality comparison
    if ground_truth is not None and all("metrics" in r for r in results.values()):
        print(f"\n{'PSF Model':<15} {'PSNR (dB)':<12} {'NRMSE':<10} {'RelErr %':<10} {'Time (s)'}")
        print("-" * 70)

        # Sort by PSNR (best first)
        for name, data in sorted(results.items(), key=lambda x: x[1]['metrics']['psnr'], reverse=True):
            m = data["metrics"]
            t = data["time"]
            marker = "🏆" if m['psnr'] == max(r['metrics']['psnr'] for r in results.values()) else "  "
            print(f"{marker} {name:<13} {m['psnr']:>8.2f}    {m['nrmse']:>6.4f}    {m['rel_error']:>6.2f}     {t:>6.1f}")

        # Show best
        best_name = max(results.items(), key=lambda x: x[1]['metrics']['psnr'])[0]
        best_psnr = results[best_name]['metrics']['psnr']
        print(f"\n🏆 Best quality: {best_name} (PSNR={best_psnr:.2f} dB)")

        if "Supplied" in results and best_name != "Supplied":
            improvement = best_psnr - results["Supplied"]["metrics"]["psnr"]
            print(f"   Improvement over supplied PSF: {improvement:+.2f} dB")
    else:
        # No metrics, just show timing
        print(f"\n{'PSF Model':<15} {'Time (s)':<12} {'Notes'}")
        print("-" * 70)

        for name, data in results.items():
            t = data["time"]
            notes = {
                "Supplied": "Original dataset PSF",
                "GL": "RI mismatch (oil→cells), compact",
                "BW": "Reference model, compact",
            }.get(name, "")
            print(f"   {name:<13} {t:>6.1f}       {notes}")

        print("\n" + "=" * 70)
        print("Note: Compare outputs visually in napari or similar viewer")

    print("\n" + "=" * 70)
    print(f"Results saved to: {output_dir}/")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

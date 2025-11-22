"""
Run JAX SHB deconvolution on synthetic dataset with three-way PSF comparison.

This script:
  - Loads synthetic dataset with ground truth
  - Tests THREE PSF models:
    1. Supplied PSF (from dataset, image-sized: 512×256×128)
    2. Gibson-Lanni PSF (generated, compact: ~15×15×25)
    3. Born-Wolf PSF (generated, compact: ~15×15×25)
  - Runs JAX SHB deconvolution with 50 iterations for each
  - Compares results against ground truth (PSNR, NRMSE, RelErr)
  - Shows which PSF model produces best reconstruction
  - Demonstrates that compact PSFs work just as well (more efficient!)

Usage (from repo root):
    python demo/run_synthetic_jax_shb.py
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
    data_dir = demo_dir / "synthetic_data"
    output_dir = demo_dir / "outputs" / "synthetic_jax_shb"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("JAX SHB DECONVOLUTION - SYNTHETIC DATASET")
    print("=" * 70)

    # Load data
    print("\nLoading synthetic dataset...")
    im_zyx = tf.imread(data_dir / "input.tif").astype(np.float32)
    ground_truth = tf.imread(data_dir / "ground-truth.tif").astype(np.float32)
    psf_supplied_zyx = tf.imread(data_dir / "psf.tif").astype(np.float32)

    # Convert to XYZ for processing
    im_xyz = np.transpose(im_zyx, (2, 1, 0))
    psf_supplied = np.transpose(psf_supplied_zyx, (2, 1, 0))
    M, N, P = im_xyz.shape

    print(f"  Image: {im_xyz.shape} (XYZ) = {im_zyx.shape} (ZYX)")
    print(f"  Ground truth: {ground_truth.shape} (ZYX)")
    print(f"  Supplied PSF: {psf_supplied.shape} (XYZ)")

    # Determine voxel sizes (typical for this dataset)
    dxy = 0.1  # 100nm lateral pixels
    dz = 0.2   # 200nm z-steps

    print(f"  Voxel size: {dxy}×{dxy}×{dz} μm")

    # Generate PSFs matching image dimensions
    print("\n" + "=" * 70)
    print("GENERATING PSFs - MATCHING IMAGE SIZE")
    print("=" * 70)

    # Common PSF parameters for synthetic dataset
    # Note: Dataset appears to use 100x/1.4NA oil immersion
    NA = 1.4
    wvl = 0.52  # 520nm (green)
    M_mag = 100  # 100x magnification

    print(f"Image dimensions: {M}×{N}×{P}")
    print(f"Generating physically-sized PSFs (memory efficient)...")

    # 0. Use supplied PSF from dataset
    print("\n0. Supplied PSF (from dataset)")
    psf_supplied_norm = psf_supplied / psf_supplied.sum()
    print(f"   Loaded: {psf_supplied.shape}, sum={psf_supplied_norm.sum():.6f}")
    print(f"   (Image-sized: {psf_supplied.shape[0]}×{psf_supplied.shape[1]}×{psf_supplied.shape[2]})")

    # 1. Gibson-Lanni PSF (physically-sized, efficient)
    print("\n1. Gibson-Lanni PSF (oil→cells)")
    psf_gl = dwpy.auto_generate_psf_gl(
        im_xyz,
        dxy=dxy, dz=dz,
        NA=NA, ni=1.515, ns=1.38, wvl=wvl, M=M_mag,
        match_image_size=False  # Physically-sized (efficient)
    )
    print(f"   Generated: {psf_gl.shape}, sum={psf_gl.sum():.6f}")
    print(f"   (Compact: {psf_gl.size / psf_supplied.size * 100:.1f}% of supplied PSF size)")
    tf.imwrite(output_dir / "PSF_GL.tif", np.transpose(psf_gl, (2, 1, 0)))

    # 2. Born-Wolf PSF (physically-sized, efficient)
    print("\n2. Born-Wolf PSF (reference)")
    psf_bw = dwpy.auto_generate_psf_bw(
        im_xyz,
        dxy=dxy, dz=dz,
        NA=NA, ni=1.515, wvl=wvl,
        match_image_size=False  # Physically-sized (efficient)
    )
    print(f"   Generated: {psf_bw.shape}, sum={psf_bw.sum():.6f}")
    print(f"   (Compact: {psf_bw.size / psf_supplied.size * 100:.1f}% of supplied PSF size)")
    tf.imwrite(output_dir / "PSF_BW.tif", np.transpose(psf_bw, (2, 1, 0)))

    # Configuration
    cfg = dwpy.DeconvolutionConfig(
        n_iter=50,
        border_quality=2,
        positivity=True,
        metric="idiv",
        use_weights=True,
        offset=0.0,
        pad_fast_fft=True,
    )

    print(f"\nConfiguration: {cfg.n_iter} iterations, SHB method, JAX backend")

    results = {}

    # Run with supplied PSF
    print("\n" + "=" * 70)
    print("DECONVOLUTION WITH SUPPLIED PSF")
    print("=" * 70)

    try:
        import jax
        print(f"JAX backend: {jax.default_backend()}")

        start = time.perf_counter()
        result_supplied = dwpy.deconvolve_fast(
            im_xyz, psf_supplied_norm,
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

        # Calculate metrics
        metrics_supplied = calculate_metrics(result_supplied_zyx, ground_truth, "\nQuality Metrics (Supplied PSF):")
        results["Supplied"] = {"time": elapsed_supplied, "metrics": metrics_supplied}

    except ImportError as e:
        print(f"✗ JAX not available: {e}")
        return 1

    # Run with Gibson-Lanni PSF
    print("\n" + "=" * 70)
    print("DECONVOLUTION WITH GIBSON-LANNI PSF")
    print("=" * 70)

    try:
        import jax
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

        # Calculate metrics
        metrics_gl = calculate_metrics(result_gl_zyx, ground_truth, "\nQuality Metrics (GL PSF):")
        results["GL"] = {"time": elapsed_gl, "metrics": metrics_gl}

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

    # Calculate metrics
    metrics_bw = calculate_metrics(result_bw_zyx, ground_truth, "\nQuality Metrics (BW PSF):")
    results["BW"] = {"time": elapsed_bw, "metrics": metrics_bw}

    # Comparison summary
    print("\n" + "=" * 70)
    print("THREE-WAY PSF COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\n{'PSF Model':<15} {'PSNR (dB)':<12} {'NRMSE':<10} {'RelErr %':<10} {'Time (s)'}")
    print("-" * 70)

    # Sort by PSNR (best first)
    for name, data in sorted(results.items(), key=lambda x: x[1]['metrics']['psnr'], reverse=True):
        m = data["metrics"]
        t = data["time"]
        marker = "🏆" if m['psnr'] == max(r['metrics']['psnr'] for r in results.values()) else "  "
        print(f"{marker} {name:<13} {m['psnr']:>8.2f}    {m['nrmse']:>6.4f}    {m['rel_error']:>6.2f}     {t:>6.1f}")

    # Show improvements
    best_name = max(results.items(), key=lambda x: x[1]['metrics']['psnr'])[0]
    best_psnr = results[best_name]['metrics']['psnr']

    print(f"\n🏆 Best quality: {best_name} (PSNR={best_psnr:.2f} dB)")

    if "Supplied" in results and best_name != "Supplied":
        improvement = best_psnr - results["Supplied"]["metrics"]["psnr"]
        print(f"   Improvement over supplied PSF: +{improvement:.2f} dB")

    print("\n" + "=" * 70)
    print(f"Results saved to: {output_dir}/")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Run JAX SHB deconvolution on synthetic dataset with GL and BW PSF comparison.

This script:
  - Loads synthetic dataset with ground truth
  - Generates both Gibson-Lanni and Born-Wolf PSFs
  - Runs JAX SHB deconvolution with 50 iterations
  - Compares results against ground truth
  - Saves outputs and quality metrics

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

    # Convert to XYZ for processing
    im_xyz = np.transpose(im_zyx, (2, 1, 0))
    M, N, P = im_xyz.shape

    print(f"  Image: {im_xyz.shape} (XYZ) = {im_zyx.shape} (ZYX)")
    print(f"  Ground truth: {ground_truth.shape} (ZYX)")

    # Determine voxel sizes (typical for this dataset)
    dxy = 0.1  # 100nm lateral pixels
    dz = 0.2   # 200nm z-steps

    print(f"  Voxel size: {dxy}×{dxy}×{dz} μm")

    # Generate PSFs matching image dimensions
    print("\n" + "=" * 70)
    print("GENERATING PSFs - MATCHING IMAGE SIZE")
    print("=" * 70)

    # Common PSF parameters for synthetic dataset
    NA = 1.4
    wvl = 0.52  # 520nm (green)
    M_mag = 60

    print(f"Image dimensions: {M}×{N}×{P}")
    print(f"Generating PSFs to match (auto-sized then padded)...")

    # 1. Gibson-Lanni PSF (auto-sized and padded to image dimensions)
    print("\n1. Gibson-Lanni PSF (oil→cells)")
    psf_gl = dwpy.auto_generate_psf_gl(
        im_xyz,
        dxy=dxy, dz=dz,
        NA=NA, ni=1.515, ns=1.38, wvl=wvl, M=M_mag,
        match_image_size=True  # Pad to match image
    )
    print(f"   Generated: {psf_gl.shape}, sum={psf_gl.sum():.6f}")
    tf.imwrite(output_dir / "PSF_GL.tif", np.transpose(psf_gl, (2, 1, 0)))

    # 2. Born-Wolf PSF (auto-sized and padded to image dimensions)
    print("\n2. Born-Wolf PSF (reference)")
    psf_bw = dwpy.auto_generate_psf_bw(
        im_xyz,
        dxy=dxy, dz=dz,
        NA=NA, ni=1.515, wvl=wvl,
        match_image_size=True  # Pad to match image
    )
    print(f"   Generated: {psf_bw.shape}, sum={psf_bw.sum():.6f}")
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
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\n{'PSF Model':<15} {'PSNR (dB)':<12} {'NRMSE':<10} {'RelErr %':<10} {'Time (s)'}")
    print("-" * 70)

    for name in ["GL", "BW"]:
        m = results[name]["metrics"]
        t = results[name]["time"]
        print(f"{name:<15} {m['psnr']:>8.2f}    {m['nrmse']:>6.4f}    {m['rel_error']:>6.2f}     {t:>6.1f}")

    # Determine winner
    if metrics_gl["psnr"] > metrics_bw["psnr"]:
        improvement = metrics_gl["psnr"] - metrics_bw["psnr"]
        print(f"\n🏆 Gibson-Lanni PSF produces better quality (+{improvement:.2f} dB)")
    elif metrics_bw["psnr"] > metrics_gl["psnr"]:
        improvement = metrics_bw["psnr"] - metrics_gl["psnr"]
        print(f"\n🏆 Born-Wolf PSF produces better quality (+{improvement:.2f} dB)")
    else:
        print(f"\n⚖️  Both PSF models produce similar quality")

    print("\n" + "=" * 70)
    print(f"Results saved to: {output_dir}/")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

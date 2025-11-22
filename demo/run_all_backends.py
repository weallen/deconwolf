"""
Run multiple deconvolution backends on the demo DAPI dataset and save outputs.

Backends covered:
  - NumPy RL/SHB
  - JAX RL/SHB (via dw_fast)
  - Numba RL/SHB
  - FFTW RL/SHB

Runs with BOTH PSF models for comparison:
  - Gibson-Lanni (GL): Accounts for RI mismatch (oil→cells)
  - Born-Wolf (BW): Simpler model for reference

Outputs are written to demo/outputs/dapi_dataset/, transposed back to ZYX
so they align with the original TIFFs in viewers like Napari.

Usage (from repo root):
    python demo/run_all_backends.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tifffile as tf

# Ensure repo root on sys.path before importing our modules
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import only what we need to avoid pulling optional deps (e.g., dask) at import time
# Import modules directly to avoid python/__init__ (which pulls dask)
import importlib.util

def _import_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

PYTHON_DIR = ROOT / "dwpy"
dw_fast = _import_module(PYTHON_DIR / "dw_fast.py", "dwpy.dw_fast")
dw_numpy = _import_module(PYTHON_DIR / "dw_numpy.py", "dwpy.dw_numpy")
psf_module = _import_module(PYTHON_DIR / "psf.py", "dwpy.psf")


def load_demo():
    demo_dir = Path(__file__).resolve().parent
    data_dir = demo_dir / "dapi_data"
    output_dir = demo_dir / "outputs" / "dapi_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    im = tf.imread(data_dir / "dapi_001.tif").astype(np.float32)
    # TIFFs are ZYX; convert to internal XYZ
    im_xyz = np.transpose(im, (2, 1, 0))

    # Common parameters for DAPI imaging
    dxy = 0.065  # microns (65nm lateral pixel size)
    dz = 0.200   # microns (200nm z-step)

    # Generate BOTH PSF models for comparison
    print("Generating PSFs for DAPI imaging (60x/1.4NA, λ=466nm)...")
    print("=" * 70)

    # 1. Gibson-Lanni PSF (accounts for RI mismatch)
    print("\n1. Gibson-Lanni PSF (oil→cells RI mismatch)")
    print("   Parameters: ni=1.515 (oil), ns=1.38 (cells)")
    psf_gl = psf_module.generate_psf_gl(
        dxy=dxy,
        dz=dz,
        xy_size=181,       # Match existing PSF size
        z_size=79,
        NA=1.4,
        ni=1.515,          # Oil immersion
        ns=1.38,           # Cellular RI
        wvl=0.466,         # DAPI emission
        M=60.0,
        ti0=150.0,
        tg=170.0,
        ng=1.515,
    )
    psf_gl_path = output_dir / "PSF_dapi_GL.tif"
    save_tif_xyz_as_zyx(psf_gl_path, psf_gl)
    print(f"   Saved: {psf_gl_path.name}")

    # 2. Born-Wolf PSF (simpler model)
    print("\n2. Born-Wolf PSF (reference)")
    print("   Parameters: ni=1.515 (oil)")
    psf_bw = psf_module.generate_psf_bw(
        dxy=dxy,
        dz=dz,
        xy_size=181,
        z_size=79,
        NA=1.4,
        ni=1.515,
        wvl=0.466,
    )
    psf_bw_path = output_dir / "PSF_dapi_BW.tif"
    save_tif_xyz_as_zyx(psf_bw_path, psf_bw)
    print(f"   Saved: {psf_bw_path.name}")

    print("\n" + "=" * 70)
    print(f"PSFs generated: shape={psf_gl.shape}")
    print(f"  GL sum: {psf_gl.sum():.6f}, BW sum: {psf_bw.sum():.6f}")
    print()

    return output_dir, im_xyz, psf_gl, psf_bw


def save_tif_xyz_as_zyx(path: Path, arr_xyz: np.ndarray) -> None:
    arr_zyx = np.transpose(arr_xyz.astype(np.float32), (2, 1, 0))
    tf.imwrite(path, arr_zyx)
    print(
        f"[wrote] {path} xyz_shape={arr_xyz.shape} zyx_shape={arr_zyx.shape} "
        f"min={arr_zyx.min():.3f} max={arr_zyx.max():.3f}"
    )


def run_numpy(im: np.ndarray, psf: np.ndarray, method: str, psf_name: str, demo_dir: Path, n_iter: int = 20) -> None:
    cfg = dw_numpy.DeconvolutionConfig(
        n_iter=n_iter,
        border_quality=2,
        positivity=True,
        metric="idiv",
        start_condition="flat",
        xycropfactor=0.001,
        use_weights=True,
        offset=5.0,
        pad_fast_fft=False,
        alphamax=1.0,
    )
    out = dw_numpy.deconvolve(im, psf, method=method, cfg=cfg)
    out_path = demo_dir / f"dw_dapi_numpy_{method}_{psf_name}.tif"
    save_tif_xyz_as_zyx(out_path, out)


def run_fast_jax(im: np.ndarray, psf: np.ndarray, method: str, psf_name: str, demo_dir: Path, n_iter: int = 20) -> None:
    try:
        import jax  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dep
        print(f"[skip] fast JAX backend not available: {exc}")
        return

    cfg = dw_numpy.DeconvolutionConfig(
        n_iter=n_iter,
        border_quality=2,
        positivity=True,
        metric="idiv",
        start_condition="flat",
        xycropfactor=0.001,
        use_weights=True,
        offset=5.0,
        pad_fast_fft=False,
        alphamax=1.0,
    )
    out = dw_fast.deconvolve_fast(im, psf, method=method, backend="jax", cfg=cfg)
    out_path = demo_dir / f"dw_dapi_fastjax_{method}_{psf_name}.tif"
    save_tif_xyz_as_zyx(out_path, out)


def run_fast_numba(im: np.ndarray, psf: np.ndarray, method: str, psf_name: str, demo_dir: Path, n_iter: int = 20) -> None:
    try:
        import numba  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dep
        print(f"[skip] fast numba backend not available: {exc}")
        return

    cfg = dw_numpy.DeconvolutionConfig(
        n_iter=n_iter,
        border_quality=2,
        positivity=True,
        metric="idiv",
        start_condition="flat",
        xycropfactor=0.001,
        use_weights=True,
        offset=0.0,
        pad_fast_fft=False,
        alphamax=1.0,
    )
    out = dw_fast.deconvolve_fast(im, psf, method=method, backend="numba", cfg=cfg)
    out_path = demo_dir / f"dw_dapi_fastnumba_{method}_{psf_name}.tif"
    save_tif_xyz_as_zyx(out_path, out)


def run_fast_fftw(im: np.ndarray, psf: np.ndarray, method: str, psf_name: str, demo_dir: Path, n_iter: int = 20) -> None:
    try:
        import pyfftw  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dep
        print(f"[skip] fast fftw backend not available: {exc}")
        return

    cfg = dw_numpy.DeconvolutionConfig(
        n_iter=n_iter,
        border_quality=2,
        positivity=True,
        metric="idiv",
        start_condition="flat",
        xycropfactor=0.001,
        use_weights=True,
        offset=0.0,
        pad_fast_fft=False,
        alphamax=1.0,
        fftw_threads=1,
    )
    out = dw_fast.deconvolve_fast(im, psf, method=method, backend="fftw", cfg=cfg)
    out_path = demo_dir / f"dw_dapi_fastfftw_{method}_{psf_name}.tif"
    save_tif_xyz_as_zyx(out_path, out)


def main():
    demo_dir, im_xyz, psf_gl, psf_bw = load_demo()

    print("\n" + "=" * 70)
    print("RUNNING DECONVOLUTION WITH BOTH PSF MODELS")
    print("=" * 70)

    # Run with Gibson-Lanni PSF
    print("\n>>> Using Gibson-Lanni PSF (GL)")
    print("-" * 70)
    run_numpy(im_xyz, psf_gl, method="rl", psf_name="GL", demo_dir=demo_dir)
    run_numpy(im_xyz, psf_gl, method="shb", psf_name="GL", demo_dir=demo_dir)
    run_fast_jax(im_xyz, psf_gl, method="rl", psf_name="GL", demo_dir=demo_dir)
    run_fast_jax(im_xyz, psf_gl, method="shb", psf_name="GL", demo_dir=demo_dir)
    run_fast_numba(im_xyz, psf_gl, method="rl", psf_name="GL", demo_dir=demo_dir)
    run_fast_numba(im_xyz, psf_gl, method="shb", psf_name="GL", demo_dir=demo_dir)
    run_fast_fftw(im_xyz, psf_gl, method="rl", psf_name="GL", demo_dir=demo_dir)
    run_fast_fftw(im_xyz, psf_gl, method="shb", psf_name="GL", demo_dir=demo_dir)

    # Run with Born-Wolf PSF
    print("\n>>> Using Born-Wolf PSF (BW)")
    print("-" * 70)
    run_numpy(im_xyz, psf_bw, method="rl", psf_name="BW", demo_dir=demo_dir)
    run_numpy(im_xyz, psf_bw, method="shb", psf_name="BW", demo_dir=demo_dir)
    run_fast_jax(im_xyz, psf_bw, method="rl", psf_name="BW", demo_dir=demo_dir)
    run_fast_jax(im_xyz, psf_bw, method="shb", psf_name="BW", demo_dir=demo_dir)
    run_fast_numba(im_xyz, psf_bw, method="rl", psf_name="BW", demo_dir=demo_dir)
    run_fast_numba(im_xyz, psf_bw, method="shb", psf_name="BW", demo_dir=demo_dir)
    run_fast_fftw(im_xyz, psf_bw, method="rl", psf_name="BW", demo_dir=demo_dir)
    run_fast_fftw(im_xyz, psf_bw, method="shb", psf_name="BW", demo_dir=demo_dir)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Outputs saved to: {demo_dir}")
    print("Compare GL vs BW results to see the effect of RI mismatch modeling!")


if __name__ == "__main__":
    sys.exit(main())

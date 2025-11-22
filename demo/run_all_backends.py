"""
Run multiple deconvolution backends on the demo DAPI dataset and save outputs.

Backends covered:
  - NumPy RL
  - NumPy SHB
  - JAX RL (via dw_fast)
  - JAX SHB (via dw_fast)
  - Numba RL/SHB
  - FFTW RL/SHB

PSF is generated using Gibson-Lanni model to account for refractive index
mismatch between oil immersion (ni=1.515) and cellular specimen (ns=1.38).

Outputs are written next to the demo data, transposed back to ZYX so they
align with the original TIFFs in viewers like Napari. Settings match the C
defaults: offset=5, Bertero weights (border_quality=2), metric=idiv.

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

    # Generate Gibson-Lanni PSF for DAPI imaging
    # Parameters for 60x/1.4NA oil immersion into cellular specimen
    print("Generating Gibson-Lanni PSF for DAPI (oil immersion into cells)...")
    print("  Parameters: 60x/1.4NA, λ=466nm, ni=1.515 (oil), ns=1.38 (cells)")

    # Get voxel sizes from image metadata or use typical values
    # Typical DAPI imaging: 65nm lateral, 200nm axial
    dxy = 0.065  # microns (65nm lateral pixel size)
    dz = 0.200   # microns (200nm z-step)

    psf_xyz = psf_module.generate_psf_gl(
        dxy=dxy,
        dz=dz,
        xy_size=51,        # Larger PSF for better coverage
        z_size=51,
        NA=1.4,            # High NA oil immersion
        ni=1.515,          # Oil immersion medium
        ns=1.38,           # Cellular refractive index
        wvl=0.466,         # DAPI emission peak (466nm)
        M=60.0,            # 60x magnification
        ti0=150.0,         # Working distance (μm)
        tg=170.0,          # Coverslip thickness (μm)
        ng=1.515,          # Coverslip RI
    )

    # Save the generated PSF for inspection
    psf_path = output_dir / "PSF_dapi_GL.tif"
    save_tif_xyz_as_zyx(psf_path, psf_xyz)
    print(f"Generated PSF saved to: {psf_path}")
    print(f"  Shape (XYZ): {psf_xyz.shape}")
    print(f"  Normalization: {psf_xyz.sum():.6f}")
    print()

    return output_dir, im_xyz, psf_xyz


def save_tif_xyz_as_zyx(path: Path, arr_xyz: np.ndarray) -> None:
    arr_zyx = np.transpose(arr_xyz.astype(np.float32), (2, 1, 0))
    tf.imwrite(path, arr_zyx)
    print(
        f"[wrote] {path} xyz_shape={arr_xyz.shape} zyx_shape={arr_zyx.shape} "
        f"min={arr_zyx.min():.3f} max={arr_zyx.max():.3f}"
    )


def save_uint16_scaled(path: Path, arr_xyz: np.ndarray) -> None:
    """Match the C output scaling: auto-scale to 16-bit and write uint16."""
    arr_zyx = np.transpose(arr_xyz.astype(np.float32), (2, 1, 0))
    maxv = float(arr_zyx.max()) if arr_zyx.size else 1.0
    scale = 65535.0 / maxv if maxv > 0 else 1.0
    arr_u16 = np.clip(arr_zyx * scale, 0, 65535).astype(np.uint16)
    tf.imwrite(path, arr_u16)
    print(
        f"[wrote] {path} (uint16 scaled) scale={scale:.6f} "
        f"zyx_shape={arr_u16.shape} min={arr_u16.min()} max={arr_u16.max()}"
    )


def run_numpy(im: np.ndarray, psf: np.ndarray, method: str, demo_dir: Path, n_iter: int = 20) -> None:
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
    out_path = demo_dir / f"dw_dapi_numpy_{method}.tif"
    save_tif_xyz_as_zyx(out_path, out)
    save_uint16_scaled(demo_dir / f"dw_dapi_numpy_{method}_u16.tif", out)


def run_fast_jax(im: np.ndarray, psf: np.ndarray, method: str, demo_dir: Path, n_iter: int = 20) -> None:
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
    out_path = demo_dir / f"dw_dapi_fastjax_{method}.tif"
    save_tif_xyz_as_zyx(out_path, out)
    save_uint16_scaled(demo_dir / f"dw_dapi_fastjax_{method}_u16.tif", out)


def run_fast_numba(im: np.ndarray, psf: np.ndarray, method: str, demo_dir: Path, n_iter: int = 20) -> None:
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
    out_path = demo_dir / f"dw_dapi_fastnumba_{method}.tif"
    save_tif_xyz_as_zyx(out_path, out)
    save_uint16_scaled(demo_dir / f"dw_dapi_fastnumba_{method}_u16.tif", out)


def run_fast_fftw(im: np.ndarray, psf: np.ndarray, method: str, demo_dir: Path, n_iter: int = 20) -> None:
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
    out_path = demo_dir / f"dw_dapi_fastfftw_{method}.tif"
    save_tif_xyz_as_zyx(out_path, out)
    save_uint16_scaled(demo_dir / f"dw_dapi_fastfftw_{method}_u16.tif", out)


def main():
    demo_dir, im_xyz, psf_xyz = load_demo()

    run_numpy(im_xyz, psf_xyz, method="rl", demo_dir=demo_dir)
    run_numpy(im_xyz, psf_xyz, method="shb", demo_dir=demo_dir)

    run_fast_jax(im_xyz, psf_xyz, method="rl", demo_dir=demo_dir)
    run_fast_jax(im_xyz, psf_xyz, method="shb", demo_dir=demo_dir)
    run_fast_numba(im_xyz, psf_xyz, method="rl", demo_dir=demo_dir)
    run_fast_numba(im_xyz, psf_xyz, method="shb", demo_dir=demo_dir)
    run_fast_fftw(im_xyz, psf_xyz, method="rl", demo_dir=demo_dir)
    run_fast_fftw(im_xyz, psf_xyz, method="shb", demo_dir=demo_dir)


if __name__ == "__main__":
    sys.exit(main())

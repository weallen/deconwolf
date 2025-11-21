"""Run multiple deconvolution backends/algorithms on the demo DAPI dataset.

This script tries the NumPy RL/SHB reference and the JAX RL/SHB variants
using the same PSF and writes outputs to the demo/ folder with informative
filenames. It mirrors the makefile settings (20 iterations, BW PSF).

Usage (from repo root):
    /Users/wea/miniforge3/envs/dwpy/bin/python util/run_deconv_variants.py

Outputs (written under demo/):
    - dw_dapi_numpy_rl.tif
    - dw_dapi_numpy_shb.tif
    - dw_dapi_jax_rl.tif
    - dw_dapi_jax_shb.tif
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tifffile as tf

# Ensure repo root on sys.path when run as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Local imports
from dwpy import dw_numpy


def load_demo():
    demo_dir = Path(__file__).resolve().parent.parent / "demo"
    im = tf.imread(demo_dir / "dapi_001.tif").astype(np.float32)
    psf = tf.imread(demo_dir / "PSF_dapi_BW_python.tif").astype(np.float32)
    return demo_dir, im, psf


def save_tif_xyz_as_zyx(path: Path, arr_xyz: np.ndarray) -> None:
    """
    Save an internal (X, Y, Z) volume in ZYX order for Napari/ITK friendliness.

    The deconvolution code works in XYZ order, while the demo TIFFs are ZYX.
    Transpose back so the outputs align with the raw stack when viewed.
    """
    arr_zyx = np.transpose(arr_xyz.astype(np.float32), (2, 1, 0))
    tf.imwrite(path, arr_zyx)
    print(
        f"[wrote] {path} xyz_shape={arr_xyz.shape} zyx_shape={arr_zyx.shape} "
        f"min={arr_zyx.min():.3f} max={arr_zyx.max():.3f}"
    )


def run_numpy(im: np.ndarray, psf: np.ndarray, method: str, demo_dir: Path, n_iter: int = 20) -> None:
    cfg = dw_numpy.DeconvolutionConfig(
        n_iter=n_iter,
        border_quality=2,
        positivity=True,
        metric="idiv",
        start_condition="flat",
        xycropfactor=0.001,
        use_weights=False,  # disable Bertero weights for parity with JAX behavior
        alphamax=10.0 if method == "shb" else dw_numpy.DeconvolutionConfig().alphamax,
    )
    out = dw_numpy.deconvolve(im, psf, method=method, cfg=cfg)
    out_path = demo_dir / f"dw_dapi_numpy_{method}.tif"
    save_tif_xyz_as_zyx(out_path, out)


def run_jax(im: np.ndarray, psf: np.ndarray, method: str, demo_dir: Path, n_iter: int = 20) -> None:
    try:
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - optional dep
        print(f"[skip] JAX not available: {exc}")
        return

    from dwpy import dw_jax

    psf_j = jnp.array(psf, dtype=jnp.float32)
    im_j = jnp.array(im, dtype=jnp.float32)
    out_j = dw_jax.run_dw(im_j, psf_j, n_iter=n_iter, border_quality=2, positivity=True, method="rl" if method == "rl" else "shb_jit", verbose=False)
    out = np.array(out_j)
    out_path = demo_dir / f"dw_dapi_jax_{method}.tif"
    save_tif_xyz_as_zyx(out_path, out)


def main():
    demo_dir, im_zyx, psf_zyx = load_demo()
    # The C and Python code treat axis order as (X,Y,Z); demo TIFFs are ZYX -> transpose
    im = np.transpose(im_zyx, (2, 1, 0))
    psf = np.transpose(psf_zyx, (2, 1, 0))

    # NumPy RL and SHB
    run_numpy(im, psf, method="rl", demo_dir=demo_dir)
    run_numpy(im, psf, method="shb", demo_dir=demo_dir)

    # JAX RL and SHB (SHB uses shb_jit path inside dw_jax.run_dw)
    run_jax(im, psf, method="rl", demo_dir=demo_dir)
    run_jax(im, psf, method="shb", demo_dir=demo_dir)


if __name__ == "__main__":
    sys.exit(main())

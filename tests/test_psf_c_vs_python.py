"""
Compare Python BW PSF against C reference implementation (dw_bw).

Generates PSFs with identical parameters using both implementations
and checks numerical agreement. The Python implementation uses simpler
numerics (5-point grid sampling + linear interp) compared to the C code
(GSL adaptive quadrature + Lanczos5 interp), so we expect modest
differences in low-signal regions.
"""
import subprocess
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest

try:
    import tifffile
except ImportError:
    tifffile = None

from dwpy.psf import generate_psf_bw

DW_BW = Path(__file__).resolve().parents[1] / "build" / "dw_bw"


def c_psf_available():
    return DW_BW.exists() and tifffile is not None


# (NA, ni, wavelength_nm, dxy_nm, dz_nm, xy_size, z_size)
CONFIGS = [
    (1.45, 1.515, 600, 130, 300, 25, 25),
    (1.4, 1.515, 520, 65, 200, 25, 25),
    (1.05, 1.33, 561, 173, 500, 17, 17),
]


def generate_c_psf(na, ni, wvl_nm, dxy_nm, dz_nm, xy_size, z_size, tmpdir):
    """Generate PSF using the C dw_bw binary."""
    outfile = str(Path(tmpdir) / "c_psf.tif")
    cmd = [
        str(DW_BW),
        "--NA", str(na),
        "--ni", str(ni),
        "--lambda", str(wvl_nm),
        "--resxy", str(dxy_nm),
        "--resz", str(dz_nm),
        "--size", str(xy_size),
        "--nslice", str(z_size),
        "--overwrite",
        outfile,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(
            f"dw_bw failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    # C writes ZYX-ordered TIFF; transpose to XYZ to match Python
    psf_zyx = tifffile.imread(outfile)
    return np.transpose(psf_zyx, (2, 1, 0))


def generate_python_psf(na, ni, wvl_nm, dxy_nm, dz_nm, xy_size, z_size):
    """Generate PSF using Python BornWolfPSF."""
    return generate_psf_bw(
        dxy=dxy_nm / 1000.0,
        dz=dz_nm / 1000.0,
        xy_size=xy_size,
        z_size=z_size,
        NA=na,
        ni=ni,
        wvl=wvl_nm / 1000.0,
        oversampling_r=17,  # match C default
        n_rho_samples=512,
        pixel_samples=5,
    )


@pytest.mark.skipif(not c_psf_available(), reason="dw_bw binary or tifffile not found")
@pytest.mark.parametrize("na,ni,wvl_nm,dxy_nm,dz_nm,xy_size,z_size", CONFIGS)
def test_bw_python_vs_c(na, ni, wvl_nm, dxy_nm, dz_nm, xy_size, z_size):
    """Compare Python BW PSF against C reference.

    Expected differences come from pixel integration method:
    - C: GSL adaptive quadrature + Lanczos5 interpolation
    - Python: 5x5 uniform grid sampling + linear interpolation

    We check agreement at multiple signal thresholds.
    """
    tmpdir = tempfile.mkdtemp(prefix="psf_cmp_")
    try:
        c_psf = generate_c_psf(na, ni, wvl_nm, dxy_nm, dz_nm, xy_size, z_size, tmpdir)
        py_psf = generate_python_psf(na, ni, wvl_nm, dxy_nm, dz_nm, xy_size, z_size)

        assert c_psf.shape == py_psf.shape

        # Normalize both to sum=1
        c_n = c_psf / c_psf.sum()
        py_n = py_psf / py_psf.sum()

        # Overall correlation
        correlation = np.corrcoef(c_n.ravel(), py_n.ravel())[0, 1]

        # Peak location must match
        c_peak = np.unravel_index(np.argmax(c_n), c_n.shape)
        py_peak = np.unravel_index(np.argmax(py_n), py_n.shape)

        # Tiered relative-error checks at different signal levels
        stats = {}
        for thresh in [0.01, 0.05, 0.20]:
            mask = c_n > thresh * c_n.max()
            if mask.any():
                rel = np.abs(c_n[mask] - py_n[mask]) / c_n[mask]
                stats[thresh] = (mask.sum(), rel.max(), rel.mean())

        print(f"\n  NA={na}, ni={ni}, wvl={wvl_nm}nm, "
              f"dxy={dxy_nm}nm, dz={dz_nm}nm, {xy_size}x{z_size}")
        print(f"  Correlation: {correlation:.8f}")
        print(f"  Peak: C={c_peak}, Python={py_peak}")
        for thresh, (n, mx, mn) in stats.items():
            print(f"  >{thresh*100:3.0f}% peak: {n:5d} vox, "
                  f"max_rel={mx:.4f}, mean_rel={mn:.4f}")

        # Assertions
        assert c_peak == py_peak, f"Peak mismatch: C={c_peak} vs Python={py_peak}"
        assert correlation > 0.998, f"Correlation too low: {correlation:.6f}"

        # At >5% peak: mean rel diff < 5%, max rel diff < 15%
        if 0.05 in stats:
            _, max_rel_5, mean_rel_5 = stats[0.05]
            assert mean_rel_5 < 0.05, f"Mean rel diff (>5% peak) too large: {mean_rel_5:.4f}"
            assert max_rel_5 < 0.15, f"Max rel diff (>5% peak) too large: {max_rel_5:.4f}"

        # At >20% peak: max rel diff < 10%
        if 0.20 in stats:
            _, max_rel_20, mean_rel_20 = stats[0.20]
            assert max_rel_20 < 0.10, f"Max rel diff (>20% peak) too large: {max_rel_20:.4f}"

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.skipif(not c_psf_available(), reason="dw_bw binary or tifffile not found")
def test_bw_symmetry_matches_c():
    """Check that Python PSF has same symmetry properties as C."""
    na, ni, wvl_nm, dxy_nm, dz_nm = 1.45, 1.515, 600, 130, 300
    xy_size, z_size = 25, 25

    tmpdir = tempfile.mkdtemp(prefix="psf_sym_")
    try:
        c_psf = generate_c_psf(na, ni, wvl_nm, dxy_nm, dz_nm, xy_size, z_size, tmpdir)
        py_psf = generate_python_psf(na, ni, wvl_nm, dxy_nm, dz_nm, xy_size, z_size)

        for psf, name in [(c_psf, "C"), (py_psf, "Python")]:
            psf_n = psf / psf.sum()
            mid_z = z_size // 2

            # z mirror symmetry
            assert np.allclose(
                psf_n[:, :, :z_size // 2],
                psf_n[:, :, -1:z_size // 2:-1],
                atol=1e-6,
            ), f"{name} PSF lacks z-symmetry"

            # xy exchange symmetry (90-degree rotation)
            plane = psf_n[:, :, mid_z]
            assert np.allclose(plane, plane.T, atol=1e-6), \
                f"{name} PSF lacks x-y exchange symmetry"

            # xy mirror symmetry
            assert np.allclose(plane, plane[::-1, :], atol=1e-6), \
                f"{name} PSF lacks x-mirror symmetry"
            assert np.allclose(plane, plane[:, ::-1], atol=1e-6), \
                f"{name} PSF lacks y-mirror symmetry"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.skipif(not c_psf_available(), reason="dw_bw binary or tifffile not found")
def test_bw_defocus_profile_vs_c():
    """Compare axial (z) intensity profile through the center pixel."""
    na, ni, wvl_nm, dxy_nm, dz_nm = 1.45, 1.515, 600, 130, 300
    xy_size, z_size = 25, 51  # more z-slices to see axial profile

    tmpdir = tempfile.mkdtemp(prefix="psf_axial_")
    try:
        c_psf = generate_c_psf(na, ni, wvl_nm, dxy_nm, dz_nm, xy_size, z_size, tmpdir)
        py_psf = generate_python_psf(na, ni, wvl_nm, dxy_nm, dz_nm, xy_size, z_size)

        cx = xy_size // 2
        c_axial = c_psf[cx, cx, :] / c_psf.sum()
        py_axial = py_psf[cx, cx, :] / py_psf.sum()

        # Axial profiles should match closely at center pixel
        # (no pixel integration needed at exact center)
        corr = np.corrcoef(c_axial, py_axial)[0, 1]
        max_rel = np.max(np.abs(c_axial - py_axial) / c_axial.max())

        print(f"\n  Axial profile correlation: {corr:.8f}")
        print(f"  Max relative diff (vs peak): {max_rel:.6f}")

        assert corr > 0.999, f"Axial profile correlation too low: {corr:.6f}"
        # Max relative diff is at far-defocus low-signal z positions;
        # the center pixel still integrates over the pixel area in C
        # (GSL quadrature) vs Python (5-point grid), giving ~8% at tails.
        assert max_rel < 0.10, f"Axial profile max rel diff too large: {max_rel:.6f}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

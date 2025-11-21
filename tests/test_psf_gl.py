"""Tests for Gibson-Lanni PSF generation."""
import numpy as np
import pytest

from dwpy import generate_psf_gl, generate_psf_bw


def test_gl_psf_basic_properties():
    """Test that GL PSF has correct basic properties."""
    psf = generate_psf_gl(
        dxy=0.11, dz=0.25,
        xy_size=25, z_size=25,
        NA=1.4, ni=1.515, ns=1.33, wvl=0.52
    )

    # Check shape
    assert psf.shape == (25, 25, 25)

    # Check dtype
    assert psf.dtype == np.float32

    # Check normalization
    assert np.isclose(psf.sum(), 1.0, rtol=1e-5)

    # Check all values are non-negative
    assert np.all(psf >= 0)

    # Check peak is near center
    peak_idx = np.unravel_index(np.argmax(psf), psf.shape)
    center = tuple(s // 2 for s in psf.shape)
    # Peak should be within a few pixels of center
    assert all(abs(p - c) < 5 for p, c in zip(peak_idx, center))


def test_gl_psf_normalization():
    """Test PSF normalization with different parameters."""
    for xy_size, z_size in [(11, 11), (25, 25), (15, 21)]:
        psf = generate_psf_gl(
            dxy=0.11, dz=0.25,
            xy_size=xy_size, z_size=z_size,
            NA=1.4, ni=1.515, ns=1.33
        )
        assert np.isclose(psf.sum(), 1.0, rtol=1e-5), \
            f"PSF not normalized for size ({xy_size}, {z_size})"


def test_gl_psf_radial_symmetry():
    """Test that PSF is approximately radially symmetric."""
    psf = generate_psf_gl(
        dxy=0.11, dz=0.25,
        xy_size=25, z_size=21,
        NA=1.4, ni=1.515, ns=1.33
    )

    # Get center slice
    center_z = psf.shape[2] // 2
    center_slice = psf[:, :, center_z]

    # Check radial symmetry by comparing opposite quadrants
    cy, cx = psf.shape[0] // 2, psf.shape[1] // 2

    # Sample points at radius r=5
    r = 5
    points = [
        center_slice[cy + r, cx],
        center_slice[cy - r, cx],
        center_slice[cy, cx + r],
        center_slice[cy, cx - r],
    ]

    # Variation should be small
    std = np.std(points)
    mean = np.mean(points)
    assert std / mean < 0.1, "PSF not radially symmetric"


def test_gl_psf_different_parameters():
    """Test GL PSF with various optical parameters."""
    # 60x/1.4NA oil immersion
    psf1 = generate_psf_gl(
        dxy=0.065, dz=0.2,
        xy_size=15, z_size=11,
        NA=1.4, M=60, ni=1.515, ns=1.33
    )
    assert psf1.shape == (15, 15, 11)
    assert np.isclose(psf1.sum(), 1.0, rtol=1e-5)

    # 40x/1.25NA water immersion
    psf2 = generate_psf_gl(
        dxy=0.11, dz=0.25,
        xy_size=15, z_size=11,
        NA=1.25, M=40, ni=1.33, ns=1.33
    )
    assert psf2.shape == (15, 15, 11)
    assert np.isclose(psf2.sum(), 1.0, rtol=1e-5)

    # PSFs should be different (different objectives)
    assert not np.allclose(psf1, psf2)


def test_gl_psf_ri_mismatch_vs_match():
    """Test that RI mismatch creates different PSF than matched RI."""
    # GL with matched RI (specimen = immersion)
    psf_matched = generate_psf_gl(
        dxy=0.11, dz=0.25,
        xy_size=15, z_size=11,
        NA=1.4, ni=1.515, ns=1.515,  # Matched
        wvl=0.52
    )

    # GL with mismatched RI (aqueous specimen)
    psf_mismatched = generate_psf_gl(
        dxy=0.11, dz=0.25,
        xy_size=15, z_size=11,
        NA=1.4, ni=1.515, ns=1.33,  # Mismatched
        wvl=0.52
    )

    # They should be noticeably different
    max_diff = np.abs(psf_matched - psf_mismatched).max()
    assert max_diff > 0.001, "RI mismatch should create aberrations"


def test_gl_psf_design_vs_actual():
    """Test design vs actual parameter mismatches."""
    # Perfect system (design = actual)
    psf_perfect = generate_psf_gl(
        dxy=0.11, dz=0.25,
        xy_size=15, z_size=11,
        NA=1.4, ni=1.515, ns=1.33,
        tg0=170, tg=170,  # Matched coverslip thickness
    )

    # Coverslip thickness mismatch
    psf_aberrated = generate_psf_gl(
        dxy=0.11, dz=0.25,
        xy_size=15, z_size=11,
        NA=1.4, ni=1.515, ns=1.33,
        tg0=170, tg=180,  # 10μm thicker coverslip
    )

    # Should create aberrations
    assert not np.allclose(psf_perfect, psf_aberrated, rtol=0.01)


def test_gl_psf_default_design_values():
    """Test that design values default to actual values."""
    # Explicitly set design = actual
    psf1 = generate_psf_gl(
        dxy=0.11, dz=0.25,
        xy_size=15, z_size=11,
        NA=1.4, ni=1.515, ns=1.33,
        ni0=1.515, tg0=170, ng0=1.515
    )

    # Let design values default
    psf2 = generate_psf_gl(
        dxy=0.11, dz=0.25,
        xy_size=15, z_size=11,
        NA=1.4, ni=1.515, ns=1.33,
        # ni0, tg0, ng0 default to ni, tg, ng
    )

    # Should be identical
    assert np.allclose(psf1, psf2)


def test_gl_psf_wavelength_dependence():
    """Test that different wavelengths produce different PSFs."""
    psf_blue = generate_psf_gl(
        dxy=0.11, dz=0.25,
        xy_size=15, z_size=11,
        NA=1.4, ni=1.515, ns=1.33,
        wvl=0.46  # Blue
    )

    psf_red = generate_psf_gl(
        dxy=0.11, dz=0.25,
        xy_size=15, z_size=11,
        NA=1.4, ni=1.515, ns=1.33,
        wvl=0.63  # Red
    )

    # Different wavelengths should give different PSFs
    assert not np.allclose(psf_blue, psf_red)

    # Red wavelength should give broader PSF
    # (peak value should be lower for red)
    assert psf_red.max() < psf_blue.max()


def test_gl_vs_bw_similar_when_matched():
    """Test that GL and BW are reasonably similar when specimen RI is matched.

    Note: They won't be identical because GL includes coverslip effects
    and other optical details that BW doesn't model.
    """
    psf_gl = generate_psf_gl(
        dxy=0.11, dz=0.25,
        xy_size=15, z_size=11,
        NA=1.4, ni=1.515, ns=1.515,  # Match specimen to immersion
        wvl=0.52
    )

    psf_bw = generate_psf_bw(
        dxy=0.11, dz=0.25,
        xy_size=15, z_size=11,
        NA=1.4, ni=1.515, wvl=0.52
    )

    # They should have same shape and normalization
    assert psf_gl.shape == psf_bw.shape
    assert np.isclose(psf_gl.sum(), psf_bw.sum(), rtol=1e-5)

    # Peaks should be in similar locations
    peak_gl = np.unravel_index(np.argmax(psf_gl), psf_gl.shape)
    peak_bw = np.unravel_index(np.argmax(psf_bw), psf_bw.shape)
    assert all(abs(p1 - p2) <= 2 for p1, p2 in zip(peak_gl, peak_bw))

    # Overall correlation should be high (> 0.9)
    correlation = np.corrcoef(psf_gl.ravel(), psf_bw.ravel())[0, 1]
    assert correlation > 0.9, f"GL and BW correlation too low: {correlation}"

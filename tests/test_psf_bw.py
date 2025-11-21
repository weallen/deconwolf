import numpy as np

from python import generate_psf_bw


def _make_psf(**kwargs):
    return generate_psf_bw(
        dxy=0.13,
        dz=0.3,
        xy_size=9,
        z_size=5,
        NA=1.2,
        ni=1.33,
        wvl=0.55,
        oversampling_r=7,
        n_rho_samples=256,
        pixel_samples=5,
        **kwargs,
    )


def test_bw_psf_normalization_and_symmetry():
    psf = _make_psf()

    assert np.isclose(psf.sum(), 1.0, rtol=5e-4, atol=5e-6)
    assert np.all(psf >= 0)
    assert np.allclose(psf, psf[::-1, :, :])
    assert np.allclose(psf, psf[:, ::-1, :])
    assert np.allclose(psf[:, :, 0], psf[:, :, -1])


def test_bw_psf_center_is_peak():
    psf = _make_psf()
    center = tuple(s // 2 for s in psf.shape)
    peak_index = np.unravel_index(np.argmax(psf), psf.shape)
    assert peak_index == center
    assert psf[center] > psf.mean()

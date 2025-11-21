import numpy as np

from dwpy import dw_numpy
from dwpy import dw_fast


def test_center_psf_padded_matches_c_shift_numpy():
    psf = np.zeros((5, 5, 5), dtype=np.float32)
    peak = (4, 1, 2)
    psf[peak] = 2.5
    psf[2, 2, 2] = 1.0  # secondary peak to ensure argmax is clear

    wshape = (7, 7, 7)
    centered = dw_numpy.center_psf_padded(psf, wshape)

    # Max should be at origin after centering
    assert centered[0, 0, 0] == centered.max()

    # Rolling back by the original max position should recover the padded PSF
    unrolled = np.roll(centered, shift=peak, axis=(0, 1, 2))
    expected = np.zeros(wshape, dtype=np.float32)
    expected[: psf.shape[0], : psf.shape[1], : psf.shape[2]] = psf
    assert np.allclose(unrolled, expected)


def test_center_psf_padded_matches_c_shift_fast_backend():
    psf = np.zeros((3, 3, 3), dtype=np.float32)
    peak = (2, 0, 1)
    psf[peak] = 5.0
    wshape = (5, 5, 5)

    centered = dw_fast._pad_psf(psf, wshape, np, lambda x: x)
    assert centered[0, 0, 0] == centered.max()

    unrolled = np.roll(centered, shift=peak, axis=(0, 1, 2))
    expected = np.zeros(wshape, dtype=np.float32)
    expected[: psf.shape[0], : psf.shape[1], : psf.shape[2]] = psf
    assert np.allclose(unrolled, expected)

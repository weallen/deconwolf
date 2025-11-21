import numpy as np
import scipy.special

try:
    from pxtools.pipeline.deconvolution.microscope_psf import MicroscopePSF
except ImportError:  # pragma: no cover - optional dependency
    MicroscopePSF = None

# models of PSFs
def nikon25x105na(xy_size=17, z_size=17, dxy=0.173, dz=0.5, na_number=1.05, wvl=0.561, tl=300.0):
    return generate_psf_custom(
         dxy=dxy, dz=dz, xy_size=xy_size, z_size=z_size, wvl=wvl, NA=na_number, tl=tl*1.0e3
    )

def nikon40x125na(xy_size=17, z_size=17, dxy=0.11, dz=0.75, NA=1.25, wvl=0.561, tl=300.0):
    return generate_psf_custom(
         dxy=dxy, dz=dz, xy_size=xy_size, z_size=z_size, M=40, NA=NA, n=1.405, wd=300, tl=tl * 1.0e3, wvl=wvl
    )

def nikon60x14na(xy_size=25, z_size=25, dxy=0.11, dz=0.25, NA=1.4, wvl=0.561, tl=200.0):
    return generate_psf_custom(
         dxy=dxy, dz=dz, xy_size=xy_size, z_size=z_size, M=40, NA=NA, n=1.51, wd=150, tl=tl * 1.0e3, wvl=wvl
    )


class BornWolfPSF:
    """Born–Wolf PSF generator (matching the C reference implementation).

    All length parameters are expressed in microns. The implementation follows
    the logic in ``src/dw_bwpsf.c``/``src/bw_gsl.c``: compute a radial Born–Wolf
    profile via numerical integration of the pupil function, interpolate that
    profile to integrate over each image pixel, enforce z-symmetry, then
    normalize the volume to sum to one.
    """

    def __init__(
        self,
        wavelength: float = 0.6,
        numerical_aperture: float = 1.45,
        immersion_ri: float = 1.515,
        oversampling_r: int = 17,
        max_rho: float = 1.0,
        n_rho_samples: int = 512,
        pixel_samples: int = 5,
    ) -> None:
        self.wavelength = float(wavelength)
        self.numerical_aperture = float(numerical_aperture)
        self.immersion_ri = float(immersion_ri)
        self.oversampling_r = int(oversampling_r)
        self.max_rho = float(max_rho)
        self.n_rho_samples = int(n_rho_samples)
        self.pixel_samples = int(pixel_samples)

    def _bw_integral(self, radius_um: float, defocus_um: float) -> float:
        """Born–Wolf integral at (r, z) returning intensity via trapezoidal rule."""

        k0 = 2.0 * np.pi / self.wavelength
        NA = self.numerical_aperture
        ni = self.immersion_ri

        rho = np.linspace(0.0, self.max_rho, self.n_rho_samples)
        bessel = scipy.special.j0(k0 * NA * radius_um * rho)
        opd = (NA ** 2) * defocus_um * (rho ** 2) / (2.0 * ni)
        phase = k0 * opd
        real_part = bessel * np.cos(phase) * rho
        imag_part = -bessel * np.sin(phase) * rho

        real_val = np.trapezoid(real_part, rho)
        imag_val = np.trapezoid(imag_part, rho)
        return float(real_val * real_val + imag_val * imag_val)

    def _radial_profile(self, radii_pix: np.ndarray, dxy: float, defocus_um: float) -> np.ndarray:
        """Compute radial intensity profile for a given z."""

        vals = np.empty_like(radii_pix, dtype=np.float64)
        for idx, r_pix in enumerate(radii_pix):
            vals[idx] = self._bw_integral(r_pix * dxy, defocus_um)
        return vals

    def _pixel_mean(
        self,
        radii_pix: np.ndarray,
        radprofile: np.ndarray,
        x0: float,
        x1: float,
        y0: float,
        y1: float,
    ) -> float:
        """Approximate pixel average by uniform sub-sampling."""

        xs = np.linspace(x0, x1, self.pixel_samples)
        ys = np.linspace(y0, y1, self.pixel_samples)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        r = np.hypot(xx, yy)
        vals = np.interp(r, radii_pix, radprofile, left=radprofile[0], right=radprofile[-1])
        return float(vals.mean())

    def generate(self, dxy: float, dz: float, xy_size: int, z_size: int) -> np.ndarray:
        """Generate the 3D Born–Wolf PSF volume.

        Parameters
        ----------
        dxy : float
            Pixel size in the lateral plane (microns).
        dz : float
            Axial pixel size (microns).
        xy_size : int
            Lateral kernel size (should be odd).
        z_size : int
            Axial kernel size (should be odd).
        """

        M = int(xy_size)
        N = int(xy_size)
        P = int(z_size)
        assert M > 0 and N > 0 and P > 0

        x0 = (M - 1.0) / 2.0
        y0 = (N - 1.0) / 2.0

        radmax = int(round(np.sqrt((M - x0) ** 2 + (N - y0) ** 2))) + 1
        radsample = self.oversampling_r
        radii_pix = np.arange((radmax + 1) * radsample + 2, dtype=np.float64) / float(radsample)

        psf = np.zeros((M, N, P), dtype=np.float64)

        n_z_half = (P + 1) // 2
        for zi in range(n_z_half):
            defocus = dz * (zi - (P - 1.0) / 2.0)
            radprofile = self._radial_profile(radii_pix, dxy=dxy, defocus_um=defocus)

            for x in range(0, int(M / 2) + 1):
                xf = M - x - 1
                for y in range(x, int(N / 2) + 1):
                    yf = N - y - 1

                    x_center = x - x0
                    y_center = y - y0

                    intensity = self._pixel_mean(
                        radii_pix,
                        radprofile,
                        x_center - 0.5,
                        x_center + 0.5,
                        y_center - 0.5,
                        y_center + 0.5,
                    )

                    psf[x, y, zi] = intensity
                    psf[x, yf, zi] = intensity
                    psf[xf, y, zi] = intensity
                    psf[xf, yf, zi] = intensity
                    psf[y, x, zi] = intensity
                    psf[yf, x, zi] = intensity
                    psf[y, xf, zi] = intensity
                    psf[yf, xf, zi] = intensity

        # z symmetry
        for zi in range(n_z_half):
            if 2 * zi < P - 1:
                psf[:, :, P - zi - 1] = psf[:, :, zi]

        psf /= psf.sum()
        return psf.astype(np.float32)


def generate_psf_bw(
    dxy: float,
    dz: float,
    xy_size: int,
    z_size: int,
    NA: float = 1.45,
    ni: float = 1.515,
    wvl: float = 0.6,
    oversampling_r: int = 17,
    max_rho: float = 1.0,
    n_rho_samples: int = 512,
    pixel_samples: int = 5,
) -> np.ndarray:
    """Generate a Born–Wolf PSF (native implementation).

    Parameters are in microns; output shape is ``(xy_size, xy_size, z_size)``
    and is normalized to sum to one.
    """

    generator = BornWolfPSF(
        wavelength=wvl,
        numerical_aperture=NA,
        immersion_ri=ni,
        oversampling_r=oversampling_r,
        max_rho=max_rho,
        n_rho_samples=n_rho_samples,
        pixel_samples=pixel_samples,
    )
    return generator.generate(dxy=dxy, dz=dz, xy_size=xy_size, z_size=z_size)


def generate_psf_custom(dxy, dz, xy_size, z_size, M=25, NA=1.05, n=1.33, wd=550, tl=300.0 * 1.0e3, wvl=0.561, ni=1.405):
    """
    Generates a 3D PSF array.
    :param dxy: voxel dimension along xy (microns)
    :param dz: voxel dimension along z (microns)
    :param xy_size: size of PSF kernel along x and y (odd integer)
    :param z_size: size of PSF kernel along z (odd integer)
        self.parameters = {
            "M": 100.0,  # magnification
            "NA": 1.4,  # numerical aperture
            "ng0": 1.515,  # coverslip RI design value
            "ng": 1.515,  # coverslip RI experimental value
            "ni0": 1.515,  # immersion medium RI design value
            "ni": 1.515,  # immersion medium RI experimental value
            "ns": 1.33,  # specimen refractive index (RI)
            "ti0": 150,  # microns, working distance (immersion medium thickness) design value
            "tg": 170,  # microns, coverslip thickness experimental value
            "tg0": 170,  # microns, coverslip thickness design value
            "zd0": 200.0 * 1.0e3,
        }  # microscope tube length (in microns).

    """

    if MicroscopePSF is None:
        raise ImportError("MicroscopePSF (pxtools) is required for generate_psf_custom")

    psf_gen = MicroscopePSF()

    # Microscope parameters.
    psf_gen.parameters["M"] = M  # magnification
    psf_gen.parameters["NA"] = NA  # numerical aperture
    psf_gen.parameters["ni0"] = ni
    psf_gen.parameters["ni"] = ni
    psf_gen.parameters["ns"] = n
    psf_gen.parameters["ti0"] = wd
    psf_gen.parameters["zd0"] = tl

    lz = (z_size) * dz
    z_offset = -(lz - 2 * dz) / 2
    pz = np.arange(0, lz, dz)

    # gLXYZParticleScan(self, dxy, xy_size, pz, normalize = True, wvl = 0.6, zd = None, zv = 0.0):
    psf_xyz_array = psf_gen.gLXYZParticleScan(dxy=dxy, xy_size=xy_size, pz=pz, zv=z_offset, wvl=wvl)

    psf_xyz_array /= psf_xyz_array.sum()

    #aprint(f"Generating PSF for parameters: {psf_gen.parameters}")

    return psf_xyz_array

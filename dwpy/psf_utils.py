"""
Utilities for PSF generation and sizing.
"""
import numpy as np
from typing import Tuple, Optional


def auto_psf_size_c_heuristic(
    dxy: float,
    dz: float,
    NA: float,
    wvl: float,
    ni: float,
    xy_size: Optional[int] = None,
    z_size: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Calculate PSF size using deconwolf C code heuristic.

    Matches the sizing logic from dw_bwpsf.c for consistency with the
    C implementation.

    Parameters
    ----------
    dxy : float
        Lateral pixel size in microns
    dz : float
        Axial pixel size in microns
    NA : float
        Numerical aperture
    wvl : float
        Wavelength in microns
    ni : float
        Immersion medium refractive index
    xy_size : int, optional
        Manual override for lateral size. If None, uses C default (181)
    z_size : int, optional
        Manual override for axial size. If None, auto-calculates using C formula

    Returns
    -------
    xy_size : int
        Lateral PSF size in pixels (odd number)
    z_size : int
        Axial PSF size in pixels (odd number)

    Notes
    -----
    The C code formula for axial size (from dw_bwpsf.c lines 424-433):
        nslice = floor(181.0 * 300.0 / resAxial / 2.0)
        P = nslice * 2 + 3

    Where:
    - 181 pixels = reference lateral PSF size
    - 300 nm = reference axial pixel size
    - 54,300 nm total = assumed imaging depth
    - Formula scales based on actual axial resolution

    Examples
    --------
    >>> # DAPI imaging: 130nm lateral, 300nm axial
    >>> xy, z = auto_psf_size_c_heuristic(
    ...     dxy=0.130, dz=0.300, NA=1.45, wvl=0.461, ni=1.512
    ... )
    >>> print(f"PSF size: {xy}x{xy}x{z}")
    PSF size: 181x181x183

    >>> # Higher resolution: 65nm lateral, 200nm axial
    >>> xy, z = auto_psf_size_c_heuristic(
    ...     dxy=0.065, dz=0.200, NA=1.4, wvl=0.52, ni=1.515
    ... )
    >>> print(f"PSF size: {xy}x{xy}x{z}")
    PSF size: 181x181x273
    """
    # Lateral size: C code default or manual override
    if xy_size is None:
        xy_size = 181  # C code default from dw_psf.c and dw_bwpsf.c

    # Axial size: C code auto-calculation formula or manual override
    if z_size is None:
        # Convert dz to nm for formula
        resAxial_nm = dz * 1000.0

        # C code formula: nslice = floor(181.0 * 300.0 / resAxial / 2.0)
        nslice = int(np.floor(181.0 * 300.0 / resAxial_nm / 2.0))
        z_size = nslice * 2 + 3  # Guarantees odd number

    # Ensure odd sizes (for Fourier symmetry)
    if xy_size % 2 == 0:
        xy_size += 1
    if z_size % 2 == 0:
        z_size += 1

    # Minimum size constraints
    xy_size = max(xy_size, 11)
    z_size = max(z_size, 11)

    return xy_size, z_size


def pad_psf_to_image_size(
    psf: np.ndarray,
    image_shape: Tuple[int, int, int],
) -> np.ndarray:
    """
    Pad or crop PSF to exactly match image dimensions.

    This is useful for FFT-based deconvolution where having PSF and image
    the same size can simplify processing (though not strictly required).

    Parameters
    ----------
    psf : np.ndarray
        PSF array, shape (X, Y, Z), dtype float32, normalized
    image_shape : tuple
        Target image shape (X, Y, Z)

    Returns
    -------
    np.ndarray
        PSF padded/cropped to image_shape, re-normalized to sum=1.0

    Examples
    --------
    >>> psf_small = np.ones((21, 21, 25), dtype=np.float32)
    >>> psf_small = psf_small / psf_small.sum()
    >>> psf_large = pad_psf_to_image_size(psf_small, (512, 256, 128))
    >>> psf_large.shape
    (512, 256, 128)
    >>> np.isclose(psf_large.sum(), 1.0)
    True
    """
    M_img, N_img, P_img = image_shape
    M_psf, N_psf, P_psf = psf.shape

    # If already the right size, return as-is
    if (M_psf, N_psf, P_psf) == (M_img, N_img, P_img):
        return psf

    # Create zero-padded array
    psf_matched = np.zeros((M_img, N_img, P_img), dtype=np.float32)

    # Calculate offsets to center the PSF
    x_offset = (M_img - M_psf) // 2
    y_offset = (N_img - N_psf) // 2
    z_offset = (P_img - P_psf) // 2

    # Calculate valid ranges for copying
    x_src_start = max(0, -x_offset)
    x_src_end = min(M_psf, M_img - x_offset)
    x_dst_start = max(0, x_offset)
    x_dst_end = x_dst_start + (x_src_end - x_src_start)

    y_src_start = max(0, -y_offset)
    y_src_end = min(N_psf, N_img - y_offset)
    y_dst_start = max(0, y_offset)
    y_dst_end = y_dst_start + (y_src_end - y_src_start)

    z_src_start = max(0, -z_offset)
    z_src_end = min(P_psf, P_img - z_offset)
    z_dst_start = max(0, z_offset)
    z_dst_end = z_dst_start + (z_src_end - z_src_start)

    # Copy PSF into centered position
    psf_matched[
        x_dst_start:x_dst_end,
        y_dst_start:y_dst_end,
        z_dst_start:z_dst_end
    ] = psf[
        x_src_start:x_src_end,
        y_src_start:y_src_end,
        z_src_start:z_src_end
    ]

    # Re-normalize
    psf_matched = psf_matched / psf_matched.sum()

    return psf_matched


def calculate_psf_size(
    dxy: float,
    dz: float,
    NA: float,
    wvl: float,
    lateral_margin: float = 3.0,
    axial_margin: float = 3.0,
) -> Tuple[int, int]:
    """
    Calculate appropriate PSF size based on optical parameters.

    The PSF size is determined by the physical extent of the point spread
    function, which depends on the numerical aperture and wavelength.

    Parameters
    ----------
    dxy : float
        Lateral pixel/voxel size in microns
    dz : float
        Axial pixel/voxel size in microns
    NA : float
        Numerical aperture
    wvl : float
        Wavelength in microns
    lateral_margin : float, optional
        Number of Airy disk radii to include laterally (default: 3.0)
        Higher values capture more of the PSF tail but increase computation
    axial_margin : float, optional
        Number of axial PSF extents to include (default: 3.0)

    Returns
    -------
    xy_size : int
        Lateral PSF size in pixels (odd number)
    z_size : int
        Axial PSF size in pixels (odd number)

    Notes
    -----
    The Airy disk radius (first zero) is approximately:
        r_airy ≈ 0.61 * λ / NA

    The axial extent (distance between first minima) is approximately:
        z_extent ≈ 2 * n * λ / NA²

    For high NA systems, these are approximate and the full PSF may extend
    beyond these values. The margin parameters control how much to capture.

    Examples
    --------
    >>> # 60x/1.4NA oil immersion, 520nm emission
    >>> xy_size, z_size = calculate_psf_size(
    ...     dxy=0.065, dz=0.2, NA=1.4, wvl=0.52
    ... )
    >>> print(f"PSF size: {xy_size}x{xy_size}x{z_size}")
    PSF size: 21x21x25

    >>> # Lower NA objective needs smaller PSF
    >>> xy_size, z_size = calculate_psf_size(
    ...     dxy=0.11, dz=0.25, NA=0.8, wvl=0.52
    ... )
    >>> print(f"PSF size: {xy_size}x{xy_size}x{z_size}")
    PSF size: 23x23x39
    """
    # Calculate theoretical PSF extents
    # Airy disk radius (first zero of Bessel function)
    r_airy = 0.61 * wvl / NA  # microns

    # Axial extent (simplified formula, assumes immersion RI ≈ NA for high NA)
    # More accurate: z_extent = 2 * n * wvl / NA^2, but n ≈ NA/sin(asin(NA/n))
    # For high NA oil (NA=1.4, n=1.515): z_extent ≈ 2*1.515*λ/1.4² ≈ 1.55*λ
    # Simplified: use 2*wvl/NA for lower NA, adjust for high NA
    if NA > 1.0:
        # High NA: use refractive index correction
        z_extent = 2.0 * 1.5 * wvl / (NA ** 2)
    else:
        # Lower NA: simplified formula
        z_extent = 2.0 * wvl / (NA ** 2)

    # Calculate PSF physical size with margins
    psf_lateral_size = 2 * lateral_margin * r_airy  # microns (diameter)
    psf_axial_size = 2 * axial_margin * z_extent    # microns (full extent)

    # Convert to pixels (must be odd)
    xy_size = int(np.ceil(psf_lateral_size / dxy))
    z_size = int(np.ceil(psf_axial_size / dz))

    # Ensure odd sizes (symmetric around center)
    if xy_size % 2 == 0:
        xy_size += 1
    if z_size % 2 == 0:
        z_size += 1

    # Minimum size constraints
    xy_size = max(xy_size, 11)  # At least 11 pixels
    z_size = max(z_size, 11)

    return xy_size, z_size


def auto_generate_psf_bw(
    im: np.ndarray,
    dxy: float,
    dz: float,
    NA: float = 1.4,
    ni: float = 1.515,
    wvl: float = 0.6,
    match_image_size: bool = False,
    **kwargs
) -> np.ndarray:
    """
    Automatically generate a Born-Wolf PSF sized appropriately for an image.

    Parameters
    ----------
    im : np.ndarray
        Image array (X, Y, Z)
    dxy : float
        Lateral pixel size in microns
    dz : float
        Axial pixel size in microns
    NA : float, optional
        Numerical aperture (default: 1.4)
    ni : float, optional
        Immersion medium refractive index (default: 1.515)
    wvl : float, optional
        Wavelength in microns (default: 0.6)
    match_image_size : bool, optional
        If True, pad PSF to exactly match image dimensions (default: False)
    **kwargs
        Additional arguments passed to calculate_psf_size() or generate_psf_bw()

    Returns
    -------
    np.ndarray
        PSF array (X, Y, Z), float32, normalized to sum=1.0

    Examples
    --------
    >>> import numpy as np
    >>> im = np.random.rand(256, 256, 40).astype(np.float32)
    >>> # Generate physically-sized PSF
    >>> psf = auto_generate_psf_bw(im, dxy=0.065, dz=0.2, NA=1.4, wvl=0.52)
    >>> psf.shape  # doctest: +SKIP
    (21, 21, 25)
    >>> # Generate PSF padded to match image
    >>> psf_matched = auto_generate_psf_bw(
    ...     im, dxy=0.065, dz=0.2, NA=1.4, wvl=0.52, match_image_size=True
    ... )
    >>> psf_matched.shape
    (256, 256, 40)
    """
    from .psf import generate_psf_bw

    # Calculate appropriate PSF size
    size_kwargs = {k: v for k, v in kwargs.items()
                   if k in ['lateral_margin', 'axial_margin']}
    xy_size, z_size = calculate_psf_size(dxy, dz, NA, wvl, **size_kwargs)

    # Generate PSF
    psf_kwargs = {k: v for k, v in kwargs.items()
                  if k not in ['lateral_margin', 'axial_margin']}
    psf = generate_psf_bw(
        dxy=dxy, dz=dz,
        xy_size=xy_size, z_size=z_size,
        NA=NA, ni=ni, wvl=wvl,
        **psf_kwargs
    )

    # Optionally pad to match image size
    if match_image_size:
        psf = pad_psf_to_image_size(psf, im.shape)

    return psf


def auto_generate_psf_gl(
    im: np.ndarray,
    dxy: float,
    dz: float,
    NA: float = 1.4,
    ni: float = 1.515,
    ns: float = 1.33,
    wvl: float = 0.6,
    M: float = 60.0,
    match_image_size: bool = False,
    **kwargs
) -> np.ndarray:
    """
    Automatically generate a Gibson-Lanni PSF sized appropriately for an image.

    Parameters
    ----------
    im : np.ndarray
        Image array (X, Y, Z)
    dxy : float
        Lateral pixel size in microns
    dz : float
        Axial pixel size in microns
    NA : float, optional
        Numerical aperture (default: 1.4)
    ni : float, optional
        Immersion medium refractive index (default: 1.515 for oil)
    ns : float, optional
        Specimen refractive index (default: 1.33 for water/cells)
    wvl : float, optional
        Wavelength in microns (default: 0.6)
    M : float, optional
        Magnification (default: 60.0)
    match_image_size : bool, optional
        If True, pad PSF to exactly match image dimensions (default: False)
    **kwargs
        Additional arguments passed to calculate_psf_size() or generate_psf_gl()

    Returns
    -------
    np.ndarray
        PSF array (X, Y, Z), float32, normalized to sum=1.0

    Examples
    --------
    >>> import numpy as np
    >>> im = np.random.rand(256, 256, 40).astype(np.float32)
    >>> # Generate PSF padded to match image
    >>> psf = auto_generate_psf_gl(
    ...     im, dxy=0.065, dz=0.2, NA=1.4, ni=1.515, ns=1.33, wvl=0.52,
    ...     match_image_size=True
    ... )
    >>> psf.shape
    (256, 256, 40)
    """
    from .psf import generate_psf_gl

    # Calculate appropriate PSF size
    size_kwargs = {k: v for k, v in kwargs.items()
                   if k in ['lateral_margin', 'axial_margin']}
    xy_size, z_size = calculate_psf_size(dxy, dz, NA, wvl, **size_kwargs)

    # Generate PSF
    psf_kwargs = {k: v for k, v in kwargs.items()
                  if k not in ['lateral_margin', 'axial_margin']}
    psf = generate_psf_gl(
        dxy=dxy, dz=dz,
        xy_size=xy_size, z_size=z_size,
        NA=NA, ni=ni, ns=ns, wvl=wvl, M=M,
        **psf_kwargs
    )

    # Optionally pad to match image size
    if match_image_size:
        psf = pad_psf_to_image_size(psf, im.shape)

    return psf


def explain_tiled_deconvolution():
    """
    Print explanation of how PSF works with tiled deconvolution.

    Returns
    -------
    str
        Explanation text
    """
    explanation = """
TILED DECONVOLUTION WITH PSF

## How It Works

When an image is too large to fit in memory, it's processed in overlapping tiles:

┌─────────────────────────────────────┐
│  Full Image (e.g., 2048×2048×100)  │
│                                     │
│  ┌────────┐                        │
│  │ Tile 1 │─┐                      │
│  │ 512×512│ │ overlap              │
│  └────────┘ │                      │
│     │  ┌────▼───┐                  │
│     └─→│ Tile 2 │                  │
│        │ 512×512│                  │
│        └────────┘                  │
└─────────────────────────────────────┘

## PSF Usage in Tiling

1. **Same PSF for all tiles**: The PSF represents the optical system,
   which is the same for the entire image.

2. **PSF can be larger than tiles**: The PSF size is determined by the
   optical parameters (NA, wavelength), NOT by the tile size.

   Example:
   - Tile size: 512×512×50 pixels
   - PSF size: 181×181×79 pixels ✓ (perfectly fine!)

3. **Border handling**: Each tile is padded based on border_quality:
   - border_quality=0: max(tile_size, psf_size)
   - border_quality=1: tile_size + psf_size//2
   - border_quality=2: tile_size + psf_size - 1 (best quality)

4. **Overlap region**: Tiles overlap by `tile_overlap` pixels to avoid
   edge artifacts. The overlap should be:
   - Minimum: PSF_size//2 (to handle convolution edges)
   - Recommended: PSF_size (for safety)
   - Default in dwpy: 50 pixels (works for most PSFs)

## Example Configuration

```python
import dwpy

# Image: 2048×2048×100, PSF: 181×181×79
cfg = dwpy.DeconvolutionConfig(
    n_iter=20,
    border_quality=2,
    tile_max_size=512,        # Each tile will be ≤512 in X,Y
    tile_overlap=100,         # Overlap = max(PSF_size, 50)
)

result = dwpy.deconvolve_tiled(large_image, psf, cfg=cfg)
```

## Memory Requirements

For each tile with padding:
- Tile size: 512×512×50
- With border_quality=2 padding: (512+181-1) × (512+181-1) × (50+79-1)
  ≈ 692×692×128 pixels
- Memory per tile: ~500 MB (float32, including FFT workspace)

## Key Points

✓ PSF size is independent of image/tile size
✓ PSF is determined by optical parameters (NA, λ, pixel size)
✓ Same PSF is used for all tiles
✓ Tiles must overlap by at least PSF_size//2
✓ Larger overlap = better edge handling but slower processing
"""
    print(explanation)
    return explanation


__all__ = [
    'auto_psf_size_c_heuristic',
    'calculate_psf_size',
    'pad_psf_to_image_size',
    'auto_generate_psf_bw',
    'auto_generate_psf_gl',
    'explain_tiled_deconvolution',
]

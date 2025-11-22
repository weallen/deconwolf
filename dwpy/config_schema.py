"""
Configuration schema for experiment-based deconvolution workflows.

This module defines dataclasses for microscope, imaging, PSF, and
deconvolution configurations that can be loaded from YAML/JSON files.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Literal


@dataclass
class MicroscopeConfig:
    """Microscope objective and optical configuration."""

    objective: str  # Human-readable name (e.g., "60x/1.45NA Oil")
    NA: float  # Numerical aperture
    ni: float  # Immersion medium refractive index
    ns: Optional[float] = None  # Specimen RI (None = auto-select PSF model)
    M: float = 60.0  # Magnification
    ti0: float = 150.0  # Working distance (μm)
    tg: float = 170.0  # Coverslip thickness (μm)
    ng: Optional[float] = None  # Coverslip RI (None = use ni)

    def __post_init__(self):
        """Set defaults for optional parameters."""
        if self.ng is None:
            self.ng = self.ni


@dataclass
class ImagingConfig:
    """Image acquisition parameters."""

    dxy: float  # Lateral pixel/voxel size (μm)
    dz: float  # Axial pixel/voxel size (μm)
    wavelength: float  # Emission wavelength (μm)


@dataclass
class PSFConfig:
    """PSF generation configuration."""

    model: Literal["auto", "bw", "born-wolf", "gl", "gibson-lanni"] = "auto"
    xy_size: Optional[int] = None  # None = auto-calculate (C heuristic)
    z_size: Optional[int] = None  # None = auto-calculate (C heuristic)
    # Override parameters for GL model
    ni0: Optional[float] = None  # Design immersion RI
    tg0: Optional[float] = None  # Design coverslip thickness
    ng0: Optional[float] = None  # Design coverslip RI


@dataclass
class DeconvolutionConfigParams:
    """Deconvolution algorithm parameters."""

    method: Literal["rl", "shb"] = "shb"
    backend: Literal["numpy", "jax", "cupy", "numba", "fftw"] = "jax"
    n_iter: int = 50
    border_quality: int = 2
    positivity: bool = True
    metric: str = "idiv"
    use_weights: bool = True
    offset: float = 0.0
    pad_fast_fft: bool = True
    alphamax: float = 1.0
    tile_max_size: Optional[int] = None
    tile_overlap: int = 50


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    name: str
    microscope: MicroscopeConfig
    imaging: ImagingConfig
    psf: PSFConfig = field(default_factory=PSFConfig)
    deconvolution: DeconvolutionConfigParams = field(default_factory=DeconvolutionConfigParams)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary (loaded from YAML/JSON)."""
        # Parse nested configs
        microscope = MicroscopeConfig(**data['microscope'])
        imaging = ImagingConfig(**data['imaging'])
        psf = PSFConfig(**data.get('psf', {}))
        deconvolution = DeconvolutionConfigParams(**data.get('deconvolution', {}))

        return cls(
            name=data['name'],
            microscope=microscope,
            imaging=imaging,
            psf=psf,
            deconvolution=deconvolution
        )


__all__ = [
    'MicroscopeConfig',
    'ImagingConfig',
    'PSFConfig',
    'DeconvolutionConfigParams',
    'ExperimentConfig',
]

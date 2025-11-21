#!/usr/bin/env python3
"""
Generate Born–Wolf PSFs for common objectives and save as TIFFs for inspection (e.g., in napari).

Defaults:
- 60x/1.4 oil
- 40x/1.4 oil
- 25x/1.05 oil

Outputs are ZYX-ordered float32 TIFFs with basic metadata.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import importlib.util

# Ensure project root (one level above this util/) is on sys.path so we can import the psf module directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load python/psf.py directly to avoid optional deps pulled in by python/__init__.py
PSF_PATH = ROOT / "python" / "psf.py"
spec = importlib.util.spec_from_file_location("psf", PSF_PATH)
psf_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(psf_mod)
generate_psf_bw = psf_mod.generate_psf_bw

try:
    import tifffile
except ImportError as exc:  # pragma: no cover
    raise SystemExit("tifffile is required for writing TIFFs. Install with `pip install tifffile`.") from exc


ObjectiveSpec = Dict[str, float]


OBJECTIVES: Dict[str, ObjectiveSpec] = {
    "nikon_60x_1.4_oil": {"NA": 1.4, "ni": 1.51, "dxy": 0.11, "dz": 0.25, "xy_size": 25, "z_size": 25},
    "nikon_40x_1.4_oil": {"NA": 1.4, "ni": 1.51, "dxy": 0.13, "dz": 0.3, "xy_size": 23, "z_size": 23},
    "nikon_25x_1.05_oil": {"NA": 1.05, "ni": 1.33, "dxy": 0.17, "dz": 0.4, "xy_size": 21, "z_size": 21},
}


def generate_and_save(name: str, spec: ObjectiveSpec, outdir: Path) -> Path:
    psf = generate_psf_bw(
        dxy=spec["dxy"],
        dz=spec["dz"],
        xy_size=int(spec["xy_size"]),
        z_size=int(spec["z_size"]),
        NA=spec["NA"],
        ni=spec["ni"],
        wvl=0.561,
        oversampling_r=7,
        n_rho_samples=512,
        pixel_samples=5,
    )
    psf = psf / psf.sum()

    # Save as ZYX for napari compatibility
    arr_zyx = np.transpose(psf, (2, 0, 1))
    outpath = outdir / f"{name}.tif"
    outdir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        outpath,
        arr_zyx.astype(np.float32),
        metadata={
            "axes": "ZYX",
            "dxy_um": spec["dxy"],
            "dz_um": spec["dz"],
            "NA": spec["NA"],
            "ni": spec["ni"],
        },
    )
    return outpath


def main(args: argparse.Namespace) -> None:
    outdir = Path(args.out)
    written = []
    for name, spec in OBJECTIVES.items():
        path = generate_and_save(name, spec, outdir)
        written.append(path)
        print(f"Wrote {path}")
    print("Done. You can open these TIFFs in napari (axes=ZYX).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Born–Wolf PSFs for common objectives.")
    parser.add_argument("--out", type=str, default="demo/psf_bw", help="Output directory for TIFFs")
    args = parser.parse_args()
    main(args)

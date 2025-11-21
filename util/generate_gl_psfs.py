#!/usr/bin/env python3
"""
Generate Gibson–Lanni PSFs for common objectives and save as TIFFs (for napari).

Objectives:
- 60x/1.4 oil
- 40x/1.4 oil
- 25x/1.05 oil

Outputs are float32, ZYX order with basic metadata.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict
import importlib.util

import numpy as np
import tifffile

# Load microscope_psf without triggering package __init__ (avoids optional deps)
ROOT = Path(__file__).resolve().parents[1]
MSPF_PATH = ROOT / "python" / "microscope_psf.py"
spec = importlib.util.spec_from_file_location("microscope_psf", MSPF_PATH)
ms_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(ms_mod)
MicroscopePSF = ms_mod.MicroscopePSF

ObjectiveSpec = Dict[str, float]

OBJECTIVES: Dict[str, ObjectiveSpec] = {
    "nikon_60x_1.4_oil": {"NA": 1.4, "ni": 1.51, "dxy": 0.11, "dz": 0.25, "xy_size": 25, "z_size": 25},
    "nikon_40x_1.4_oil": {"NA": 1.4, "ni": 1.51, "dxy": 0.13, "dz": 0.30, "xy_size": 23, "z_size": 23},
    "nikon_25x_1.05_oil": {"NA": 1.05, "ni": 1.33, "dxy": 0.17, "dz": 0.40, "xy_size": 21, "z_size": 21},
}


def generate_gl_psf(spec: ObjectiveSpec) -> np.ndarray:
    mp = MicroscopePSF()
    mp.parameters["NA"] = spec["NA"]
    mp.parameters["ni"] = spec["ni"]
    mp.parameters["ns"] = spec["ni"]

    z_size = int(spec["z_size"])
    pz = (np.arange(z_size) - (z_size - 1) / 2.0) * spec["dz"]
    psf = mp.gLXYZParticleScan(
        dxy=spec["dxy"],
        xy_size=int(spec["xy_size"]),
        pz=pz,
        normalize=True,
        wvl=0.561,
    )
    psf = psf / psf.sum()  # ensure unit energy
    return psf.astype(np.float32)  # shape (z, y, x)


def main(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for name, spec in OBJECTIVES.items():
        psf = generate_gl_psf(spec)
        outpath = outdir / f"{name}_gl.tif"
        tifffile.imwrite(
            outpath,
            psf,
            metadata={
                "axes": "ZYX",
                "dxy_um": spec["dxy"],
                "dz_um": spec["dz"],
                "NA": spec["NA"],
                "ni": spec["ni"],
                "model": "Gibson-Lanni",
            },
        )
        print(f"Wrote {outpath}")
    print("Done. Open these ZYX TIFFs in napari.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Gibson–Lanni PSFs for common objectives.")
    parser.add_argument("--out", type=str, default="demo/psf_gl", help="Output directory")
    args = parser.parse_args()
    main(Path(args.out))

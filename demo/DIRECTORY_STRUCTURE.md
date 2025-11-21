# Demo Directory Structure

```
demo/
├── 📁 dapi_data/                    # DAPI test dataset
│   ├── dapi_001.tif                 # Input DAPI image
│   ├── PSF_dapi.tif                 # PSF for DAPI
│   └── PSF_dapi_BW_python.tif       # Alternative PSF
│
├── 📁 synthetic_data/               # Synthetic test dataset with ground truth
│   ├── input.tif                    # Blurred input (512×256×128)
│   ├── psf.tif                      # Point spread function
│   └── ground-truth.tif             # True image (for quality metrics)
│
├── 📁 outputs/                      # All outputs organized here
│   ├── dapi_dataset/                # DAPI benchmark & run_all outputs
│   │   ├── dw_c_benchmark.tif
│   │   ├── dw_dapi_numpy_rl.tif
│   │   ├── dw_dapi_numpy_shb.tif
│   │   └── ... (all backend outputs)
│   ├── synthetic_dataset/           # Synthetic benchmark outputs
│   │   ├── output_c.tif
│   │   ├── output_numpy.tif
│   │   ├── output_jax.tif
│   │   └── output_fftw.tif
│   └── benchmarks/                  # Benchmark result summaries
│       ├── dapi_dataset_results.txt
│       └── synthetic_dataset_results.txt
│
├── 📄 benchmark_all.py              # DAPI speed benchmark
├── 📄 benchmark_new_dataset.py      # Synthetic quality benchmark ⭐
├── 📄 run_all_backends.py           # Generate all backend outputs
├── 📄 README.md                     # Script documentation
├── 📄 DIRECTORY_STRUCTURE.md        # This file
│
└── 📁 Utility folders
    ├── psf_bw/                      # Born-Wolf PSF library
    ├── psf_gl/                      # Gibson-Lanni PSF library
    └── scripts/                     # Utility scripts
```

## Quick Reference

### Run Benchmarks:
```bash
# Speed test on DAPI data
python demo/benchmark_all.py

# Quality test on synthetic data (with ground truth)
python demo/benchmark_new_dataset.py

# Generate all outputs
python demo/run_all_backends.py
```

### Find Results:
- **Speed benchmarks**: `outputs/benchmarks/*.txt`
- **DAPI outputs**: `outputs/dapi_dataset/`
- **Synthetic outputs**: `outputs/synthetic_dataset/`

### Input Data:
- **DAPI**: `dapi_data/`
- **Synthetic**: `synthetic_data/`

All scripts automatically save to correct output directories!

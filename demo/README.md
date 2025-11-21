# Demo Scripts Overview

## Directory Structure
```
demo/
├── synthetic_data/          # Synthetic test dataset with ground truth
│   ├── input.tif           # Blurred input image
│   ├── psf.tif             # Point spread function
│   └── ground-truth.tif    # True unblurred image
├── outputs/
│   ├── dapi_dataset/       # Outputs from DAPI dataset benchmarks
│   ├── synthetic_dataset/  # Outputs from synthetic dataset benchmarks
│   └── benchmarks/         # Benchmark result summaries
└── *.py                    # Benchmark scripts
```

## Available Scripts

### 1. `benchmark_all.py`
**Purpose**: Comprehensive benchmark on DAPI dataset
**Dataset**: `dapi_001.tif` + `PSF_dapi.tif`
**Tests**: C, NumPy, JAX, Numba, FFTW
**Outputs**: Saved to `outputs/dapi_dataset/`

**Run**:
```bash
python demo/benchmark_all.py
```

**What it shows**:
- Execution time for each backend
- Speed comparisons (relative to NumPy)
- Accuracy comparisons (backends vs C reference)

---

### 2. `benchmark_new_dataset.py` ⭐
**Purpose**: Benchmark on synthetic dataset WITH GROUND TRUTH
**Dataset**: `synthetic_data/input.tif` + `synthetic_data/psf.tif`
**Ground Truth**: `synthetic_data/ground-truth.tif`
**Tests**: C, NumPy, JAX, Numba, FFTW
**Outputs**: Saved to `outputs/synthetic_dataset/`

**Run**:
```bash
python demo/benchmark_new_dataset.py
```

**What it shows**:
- Execution time for each backend
- Speed comparisons
- **Reconstruction quality metrics** (PSNR, NRMSE, Relative Error)
- Which backend produces the best reconstruction

---

### 3. `run_all_backends.py`
**Purpose**: Run all Python backends and save outputs
**Dataset**: `dapi_001.tif` + `PSF_dapi.tif`
**Tests**: NumPy (RL & SHB), JAX (RL & SHB), Numba, FFTW
**Outputs**: Saved to `outputs/dapi_dataset/`

**Run**:
```bash
python demo/run_all_backends.py
```

**What it saves**:
- Float32 outputs: `dw_dapi_numpy_rl.tif`, `dw_dapi_fastjax_shb.tif`, etc.
- Uint16 outputs: `dw_dapi_numpy_rl_u16.tif`, etc.

---

## Metrics Explained

### Speed Metrics:
- **Time**: Wall-clock execution time
- **Speedup**: Relative to NumPy baseline (higher is better)

### Reconstruction Quality Metrics (vs Ground Truth):
- **PSNR** (Peak Signal-to-Noise Ratio): Higher is better (dB scale)
  - >40 dB: Excellent
  - 30-40 dB: Good
  - 20-30 dB: Acceptable
  - <20 dB: Poor

- **NRMSE** (Normalized Root Mean Square Error): Lower is better (0-1 scale)
  - <0.05: Excellent
  - 0.05-0.10: Good
  - 0.10-0.20: Acceptable
  - >0.20: Poor

- **RelErr** (Relative Error): Lower is better (percentage)
  - <5%: Excellent
  - 5-10%: Good
  - 10-20%: Acceptable
  - >20%: Poor

---

## Quick Start

**For speed comparison**:
```bash
python demo/benchmark_all.py
```

**For quality assessment** (with ground truth):
```bash
python demo/benchmark_new_dataset.py
```

**Expected Results**:
- JAX: ~3-4x faster than NumPy, similar quality
- FFTW: ~same speed as NumPy
- NumPy: Most accurate (0.0005% error vs C)
- C: Slower but reference implementation

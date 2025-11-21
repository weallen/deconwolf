#!/bin/bash
# Cleanup script for demo directory

echo "Cleaning up demo directory..."

# Remove old test outputs
echo "Removing old test outputs..."
rm -f dw_dapi_*.tif
rm -f dw_dapi_*_u16.tif
rm -f dw_dapi_numpy_rl_offset0.tif
rm -f dw_c_benchmark.tif

# Remove old log files (keep only the latest)
echo "Removing old log files..."
rm -f dw_dapi_001.tif.log.txt
rm -f dw_dapi_001_float.tif.log.txt
rm -f dw_dapi_001_offset5.tif.log.txt
rm -f dw_c_benchmark.tif.log.txt

# Remove Python cache
echo "Removing __pycache__..."
rm -rf __pycache__/

# Remove old test files
echo "Removing old test scripts..."
rm -f compare_offset0.py
rm -f test_auto.py
rm -f test_backends.py
rm -f test_numpy.py
rm -f npy_test.txt
rm -f testfile.txt

# Remove old benchmark files (keep benchmark_new_dataset.py and benchmark_all.py)
echo "Removing old benchmark scripts..."
rm -f benchmark_backends.py
rm -f quick_benchmark.py
rm -f benchmark_results.txt

echo ""
echo "Cleanup complete!"
echo ""
echo "Remaining files:"
echo "  Input data: dapi_001.tif, input.tif, PSF_dapi.tif, psf.tif"
echo "  Scripts: benchmark_all.py, benchmark_new_dataset.py, run_all_backends.py"
echo "  Results: benchmark_summary.txt, output_c.tif (if exists)"
echo "  Dirs: psf_bw/, psf_gl/, scripts/, synthetic_data/"

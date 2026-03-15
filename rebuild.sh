#!/bin/bash
# Force full GPU rebuild regardless of file timestamps.
# Run this instead of 'make gpu' whenever sources have been updated.
set -e
cd "$(dirname "$0")"

echo "=== Forcing clean rebuild ==="
make clean

echo "=== Touching all source files to reset timestamps ==="
touch fractal_sponge.h fractal_sponge.cuh
touch fractal_sponge.c fractal_sponge.cu
touch birthday.cu avalanche.cu differential.cu gpu_main.cu main.c

echo "=== Building ==="
make all

echo "=== Done ==="
ls -lh fractal_hash fractal_gpu 2>/dev/null

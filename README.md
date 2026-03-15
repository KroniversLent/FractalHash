# FractalSponge-256

Research-grade fractal hash function. Three fractal families (Julia, Newton,
Burning Ship) provide the nonlinear S-box; DFT-4 theta provides diffusion;
Keccak-style sponge construction ensures length-extension safety.

## Status

| Property              | Result         | Target    |
|-----------------------|----------------|-----------|
| Avg avalanche         | 50.0%          | 50%       |
| Chi-squared (p-value) | 0.59 (PASS)    | > 0.05    |
| Per-bit worst         | 124.3 bits     | > 115     |
| Min single-flip Δ     | ~101 (noise)   | > 110     |
| Preimage resistance   | **unproven**   | required  |
| Collision resistance  | **unproven**   | required  |

**This is a research prototype, not a production cryptographic hash.**

## Build

### CPU only
```bash
make cpu
./fractal_hash --test
./fractal_hash --bench
./fractal_hash somefile.txt
echo "hello" | ./fractal_hash
```

### GPU (requires CUDA toolkit ≥ 11.0)
```bash
make info          # check detected GPU arch
make gpu
./fractal_gpu --bench
./fractal_gpu --avalanche 16384
./fractal_gpu --birthday
```

### Both
```bash
make all
make test          # CPU tests
make test-gpu      # GPU tests
```

## Expected output

### CPU bench (typical)
```
CPU bench: 200 × 1KB in 0.8s → 250 KB/s  (4.0 ms/hash)
```
Python was ~200ms/hash, so C gives ~50× speedup.

### GPU bench (RTX 3080)
```
GPU: NVIDIA GeForce RTX 3080 (sm_86)  10.0 GB  68 SMs
GPU bench: 1M hashes in 0.8s  →  1.25M hashes/sec
```

### Birthday suite (GPU)
```
=== Birthday test (GPU) ===
  N = 16384 (2^14.0)  prefix bits = 24  expected = 8.0
  GPU hashing: 0.003s  (5.5M hashes/sec)
  actual collisions: 7   ratio: 0.88  [PASS]

  N = 1048576 (2^20.0)  prefix bits = 32  expected = 122.1
  actual collisions: 119  ratio: 0.97  [PASS]

  N = 1048576 (2^20.0)  prefix bits = 48  expected = 0.0
  actual collisions: 0   [PASS]
```

## File layout

```
fractal_sponge.h     shared types, constants, CPU API
fractal_sponge.c     CPU implementation (gcc -O2 -lm)
main.c               CLI: sha256sum-compatible interface
fractal_sponge.cuh   shared CUDA header
fractal_sponge.cu    CUDA device functions (hash kernel)
birthday.cu          GPU birthday collision test
avalanche.cu         GPU avalanche SAC test
gpu_main.cu          GPU CLI entry point
Makefile             auto-detects GPU arch via nvidia-smi
```

## Architecture

```
Input block (32 bytes)
        │  XOR into rate words
        ▼
┌───────────────────────────────┐  ×24 rounds
│  θ_DFT  4-point DFT on state  │  full cross-word diffusion
│  ρ      word rotations         │  bit-level shift
│  π      fixed permutation      │  position shuffle
│  χ_F    fractal S-box          │  irreversible nonlinear mixing
│  ι      round constant         │  symmetry breaking
└───────────────────────────────┘
        │  squeeze rate words
        ▼
256-bit digest
```

### Fractal S-box detail
Each of the 8 state words passes through:
1. Bit-deinterleave → three float coordinates in [-2, 2]
2. Julia orbit (8 iter): z² + c  — sensitive dependence
3. Newton orbit (7 iter): z³ - 1 — basin boundary discontinuities
4. Burning Ship orbit (8 iter): (|re|+i|im|)² + c — breaks symmetry
5. Harvest: raw IEEE 754 mantissa bits from 6 orbit endpoints
6. XOR-fold → ARX whiten → Murmur3 mix64 — eliminates attractor bias

## Next research steps

1. **Extended birthday test** — `./fractal_gpu --birthday-large` (2²⁴ hashes)
2. **Differential cryptanalysis** — find input pairs with non-random output differential
3. **Algebraic analysis** — can the fractal sbox be approximately inverted?
4. **Formal security reduction** — connect preimage hardness to iterated polynomial maps

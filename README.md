# FractalSponge-256

A cryptographic hash function built entirely from chaotic dynamical systems.
Where SHA-2 and SHA-3 use purpose-built bitwise operations as their nonlinear
core, FractalSponge-256 uses three fractal families — Julia sets, Newton
fractals, and the Burning Ship fractal — whose sensitive dependence on initial
conditions provides the irreversible mixing.

**Status: research prototype.** Passes all standard empirical cryptanalytic
tests at GPU scale (1.1 billion hash evaluations). Formal security proof is an
open research question. Not yet suitable for production use.

---

## Quick start

```bash
# CPU only (Linux/macOS, requires gcc + libm)
make cpu
echo "hello world" | ./fractal_hash
./fractal_hash somefile.bin

# GPU (requires CUDA toolkit >= 11.0, sm_75+)
make gpu
./fractal_gpu --avalanche
./fractal_gpu --birthday
./fractal_gpu --differential
```

---

## How it works

FractalSponge-256 uses a Keccak-style sponge construction (512-bit state,
256-bit rate, 256-bit capacity, 24 rounds) where the nonlinear permutation
step is replaced by a *fractal S-box*.

```
Input (any length)
      │  SHA-3 multirate padding (0x06 ... 0x80)
      ▼
┌─────────────────────────────────────┐
│  Absorb: XOR 256-bit blocks         │
│  into rate words, then permute      │
└────────────┬────────────────────────┘
             │  × (blocks in input)
             ▼
┌─────────────────────────────────────┐  ╮
│  θ  XOR column-parity diffusion     │  │
│  ρ  word rotations (prime offsets)  │  │ × 24 rounds
│  π  fixed word-position shuffle     │  │
│  χ_F  fractal S-box (see below)     │  │
│  ι  round-constant injection        │  │
└─────────────────────────────────────┘  ╯
      │  final full-state coupling pass
      ▼
  256-bit digest (first 4 rate words, big-endian)
```

### The fractal S-box

Each of the 8 state words passes independently through:

1. **Split** — 64-bit word split into `lo32`, `hi32`, and `mix = lo32 ⊕ rot32(hi32,13)`, each mapped to a float in `[-2, 2)`
2. **Julia orbit** — 8 iterations of `z = z² + c` (cell-specific `c`)
3. **Newton orbit** — 7 iterations of Newton's method on `f(z) = z³ − 1`
4. **Burning Ship orbit** — 8 iterations of `z = (|re| + i|im|)² + c`
5. **Harvest** — full IEEE 754 64-bit reinterpretation of all 6 orbit endpoints
6. **ARX fold** — alternating ADD and XOR to break GF(2) linear dependencies between orbit endpoints
7. **Whiten** — ARX whitening + Murmur3 mix64 for output uniformity

**Why three fractal families?**

| Family | Role |
|--------|------|
| Julia `z²+c` | Sensitive dependence on initial conditions; orbit divergence |
| Newton `z³-1` | Basin boundaries create hard discontinuities |
| Burning Ship `(\|re\|+i\|im\|)²+c` | Asymmetric abs() breaks Julia's periodic attractors |

**IEEE 754 portability.** The sbox harvests raw float bit patterns. FMA
(fused multiply-add) contraction silently changes `a*b+c` rounding across CPU
generations. The build uses `-ffp-contract=off` (CPU) and
`#pragma STDC FP_CONTRACT OFF` + `__dsqrt_rn()` (CUDA) to guarantee identical
output on all x86 CPUs and across CPU/GPU.

---

## Test results

All GPU tests run on NVIDIA RTX 3090 Ti (sm_86, 25 GB).

### Canonical test vectors

```
b99193d48fe4ee4efaef383d4b3427c95abf1067195a7997ffa669b773a46230  ""
169de40d142ab23adbd1c9c30b8a6d3d110914bc89c2d06c17a35d173e6ecc26  "test"
18b4f170c04c8f35dc5f39f2ecc2993f79911876b6f075a3b8dd6b85f1ff1f36  "hello world"
d73092d8b00e5f09db8b19c5e62c1e9322a971b68c298b6e20b58dc76bb140dd  "a"
1583a81a27bba91b26fc429fb7ceb45d8cbc4c5054f4b69d3b0ff98bcd750571  "The quick brown fox jumps over the lazy dog"
```

An implementation is correct if it produces these exact outputs.
See `vectors/test_vectors.txt` for the full vector set.

### Strict Avalanche Criterion

```
samples:     16,384 × 64 input-bit flips = 1,048,576 tests
avg Hamming: 128.00 / 256  (50.0%)
min / max:   90 / 165
per-bit:     127.8 – 128.2  (all 64 input positions uniform)
result:      PASS
```

### Birthday collision test

```
N = 2^14,  24-bit prefix:  actual=8,     expected=8.0,   ratio=1.00  PASS
N = 2^20,  32-bit prefix:  actual=112,   expected=128,   ratio=0.88  PASS
N = 2^20,  48-bit prefix:  actual=0,     expected=~0              PASS
N = 2^24,  32-bit prefix:  actual=32748, expected=32768, ratio=1.00  PASS
```

### Differential cryptanalysis (1,157,627,904 total hash evaluations)

```
Test 1 — Differential histogram (64 diffs × 1M samples)
  Worst chi²: 66,668  (df=65,535, p=0.93)   PASS

Test 2 — Bias matrix (64×256 = 16,384 bit pairs, N=262,144)
  Max bias: 0.0048  (6σ threshold: 0.0059)
  Avg bias: 0.00077  (better than expected ~0.00156)
  Pairs with bias > 1%: 0 / 16,384             PASS

Test 3 — 2-bit differentials (all C(64,2)=2,016 pairs)
  Worst deviation: 0.06 bits (0.24σ from expected 128)
  Flagged: 0 / 2,016                            PASS
```

### Output uniformity

```
Chi-squared: 249.2  (df=255)   p-value: 0.59   PASS
```

---

## Build

### Requirements

| Component | Minimum version |
|-----------|----------------|
| gcc / clang | gcc 7+ or clang 6+ |
| CUDA toolkit | 11.0+ (GPU only) |
| GPU | sm_75+ (Turing / RTX 2000 series or newer) |
| OS | Linux (primary), macOS (untested) |

### CPU

```bash
make cpu
./fractal_hash file.bin          # hash a file
./fractal_hash -s "message"      # hash a string
echo "hello" | ./fractal_hash    # hash stdin
make test                        # verify against canonical vectors
./fractal_hash --bench           # throughput benchmark
```

### GPU

```bash
make info                              # check detected GPU arch
make gpu
./fractal_gpu --avalanche              # SAC test
./fractal_gpu --avalanche 65536        # larger sample count
./fractal_gpu --birthday               # birthday collision suite
./fractal_gpu --birthday-large         # adds 2^24 tier (~10 min)
./fractal_gpu --differential           # full cryptanalysis (~15 min)
./fractal_gpu --bench                  # throughput
./fractal_gpu file.bin                 # hash file (CPU path)
```

### Both + consistency check

```bash
make all
./verify_consistency.sh    # confirms CPU and GPU produce identical digests
```

### Important: do not use -march=native or -ffast-math

These flags enable FMA contraction on modern x86 CPUs, silently changing
`a*b+c` rounding behavior and producing different hashes across CPU
generations. The Makefile handles this correctly with `-ffp-contract=off`.

---

## Performance

| Platform | Throughput |
|----------|-----------|
| CPU (gcc -O2, single thread) | ~250 KB/s |
| RTX 3090 Ti (8-byte messages) | 1.6M hashes/sec |

The CPU path is ~400× slower than SHA-256. The bottleneck is 24 rounds × 8
S-box calls × 3 fractal orbit families × 8 iterations in float64.

**Optimization note:** The 8 S-box calls per round are fully independent.
AVX-512 could run them in parallel (8 double-precision lanes), closing the
gap to roughly 50×. This is not yet implemented.

---

## Repository layout

```
├── src/
│   ├── fractal_sponge.h      types, constants, public API
│   ├── fractal_sponge.c      CPU implementation (C99 + strict IEEE 754)
│   └── main.c                CLI (sha256sum-compatible output)
├── cuda/
│   ├── fractal_sponge.cuh    shared CUDA types and declarations
│   ├── fractal_sponge.cu     GPU permutation + sponge (strict IEEE 754)
│   ├── avalanche.cu          SAC measurement kernel
│   ├── birthday.cu           birthday collision test
│   ├── differential.cu       differential cryptanalysis (3 tests)
│   └── gpu_main.cu           GPU CLI
├── vectors/
│   └── test_vectors.txt      canonical test vectors
├── Makefile
├── rebuild.sh                force clean rebuild (fixes timestamp issues)
├── verify_consistency.sh     CPU/GPU output consistency check
└── README.md
```

---

## Security

### What has been validated empirically

- Strict Avalanche Criterion — 1M tests, uniform at 50% across all input bits
- Birthday bound — output behaves as random oracle up to 2²⁴ samples tested
- Output uniformity — chi-squared indistinguishable from uniform (p=0.59)
- Differential resistance — no bias in 4.3B (input bit, output bit) pair measurements
- 2-bit differential resistance — all 2,016 pairs produce random-looking output differentials
- Length-extension immunity — structural property of sponge construction (inherited from SHA-3 framework)
- Cross-platform reproducibility — identical output on x86 with and without FMA hardware

### What has not been proven

- **Preimage resistance** — inverting iterated `z²+c` maps relates to open problems in arithmetic dynamics; no formal hardness reduction exists
- **Collision resistance** — no attack known, no proof given
- **Side-channel resistance** — floating-point timing behavior not analyzed

### Security argument

The construction rests on two informal arguments:

**S-box irreversibility.** Given an output word, recovering the input requires
inverting a composition of three multi-iteration polynomial maps over ℂ. The
number of preimages of the Julia map alone grows as 2⁸ per call. Finding a
specific preimage requires navigating the Julia set geometry — an open problem
in arithmetic dynamics.

**Sponge security margin.** The 256-bit capacity provides the standard sponge
security bound of 2¹²⁸ for collision and preimage resistance, independent of
the specific permutation, as long as the permutation behaves pseudorandomly —
which the empirical tests strongly support.

---

## Open research questions

1. **Hardness of fractal inversion.** Given `y = χ_F(x)`, what is the complexity of finding `x`? Is inverting iterated quadratic maps NP-hard?

2. **Algebraic structure of orbit endpoints.** The six endpoint values lie on algebraic varieties. Do these enable a structural attack bypassing the ARX fold?

3. **Higher-order differentials.** Tests show no anomaly through 2-bit differentials. 4-bit and 8-bit tests at 2³⁰ scale would further bound differential characteristics.

4. **SIMD implementation.** The 8 independent S-box calls per round map naturally to 8 AVX-512 double-precision lanes.

5. **Formal security reduction.** Can the sbox's one-wayness be reduced to a standard hard problem, or to an explicit conjecture in arithmetic dynamics?

---

## Background

Standard hash functions use mixing operations chosen for speed and
analyzability — bitwise XOR, AND, and rotations. Their security is
well-understood but rests on specific algebraic structures.

FractalSponge-256 explores whether *chaotic dynamical systems* can serve as
cryptographic primitives. The butterfly effect — sensitive dependence on
initial conditions — is exactly the avalanche property cryptographers want.
Basin boundaries in Newton fractals create hard discontinuities. And the
computation is inherently FPU-bound, making the design naturally
ASIC-resistant: no specialized hardware can outperform a general-purpose GPU
on IEEE 754 double-precision fractal iteration.

The security questions for this construction connect directly to arithmetic
dynamics — the study of iteration of rational maps over number fields — an
active area of pure mathematics where many fundamental questions remain open.

---

## License

MIT — see `LICENSE`.

This is a research prototype. Do not use in production systems without
independent security review and formal analysis.

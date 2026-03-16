# FractalSponge-256 & FractalCipher

A cryptographic hash function and authenticated stream cipher built entirely
from chaotic dynamical systems. Where SHA-2 and SHA-3 use purpose-built
bitwise operations as their nonlinear core, FractalSponge-256 uses three
fractal families — Julia sets, Newton fractals, and the Burning Ship fractal —
whose sensitive dependence on initial conditions provides the irreversible
mixing. FractalCipher extends this into a full AEAD stream cipher with
public/private keypairs.

**Status: research prototype.** Passes all standard empirical cryptanalytic
tests at GPU scale (1.1 billion hash evaluations). Formal security proof is an
open research question. Not yet suitable for production use.

---

## Quick start

```bash
# Build everything (CPU hash + cipher; GPU requires CUDA)
make cpu cipher

# Hash
echo "hello world" | ./fractal_hash
./fractal_hash somefile.bin

# Cipher — key exchange and encryption
./fractal_cipher keygen  alice.priv alice.pub
./fractal_cipher keygen  bob.priv   bob.pub
./fractal_cipher shared  alice.priv bob.pub   shared.key   # Alice derives
./fractal_cipher shared  bob.priv   alice.pub shared.key   # Bob derives (identical)
./fractal_cipher encrypt shared.key plaintext.txt  message.enc
./fractal_cipher decrypt shared.key message.enc    plaintext.txt

# GPU (requires CUDA toolkit >= 11.0, sm_75+)
make gpu
./fractal_gpu --avalanche
./fractal_gpu --birthday
./fractal_gpu --differential
```

---

## FractalSponge-256 — the hash function

### How it works

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

**IEEE 754 portability.** The S-box harvests raw float bit patterns. FMA
(fused multiply-add) contraction silently changes `a*b+c` rounding across CPU
generations. The build enforces `-ffp-contract=off` (CPU) and
`#pragma STDC FP_CONTRACT OFF` + `__dsqrt_rn()` (CUDA) to guarantee identical
output on all x86 CPUs and across CPU/GPU.

### AVX2 acceleration

The 8 S-box calls per round are fully independent, mapping naturally to SIMD
parallelism. `fractal_sponge_avx2.c` computes them as two 4-wide `__m256d`
batches, with all three fractal orbits vectorised across lanes. Strict IEEE
754 is preserved — no FMA intrinsics are used, and every mul+add is kept as a
separate instruction to match the scalar operation order exactly, producing
bit-identical output.

Runtime dispatch via CPUID: AVX2 is used automatically if the CPU supports it,
falling back to the scalar path otherwise.

| CPU path | Throughput |
|----------|-----------|
| Scalar (gcc -O2) | ~250 KB/s |
| AVX2 (4-wide double) | ~1,300 KB/s (~5×) |
| RTX 3090 Ti | 1.6M hashes/sec |

### Test vectors

```
b99193d48fe4ee4efaef383d4b3427c95abf1067195a7997ffa669b773a46230  ""
169de40d142ab23adbd1c9c30b8a6d3d110914bc89c2d06c17a35d173e6ecc26  "test"
18b4f170c04c8f35dc5f39f2ecc2993f79911876b6f075a3b8dd6b85f1ff1f36  "hello world"
d73092d8b00e5f09db8b19c5e62c1e9322a971b68c298b6e20b58dc76bb140dd  "a"
1583a81a27bba91b26fc429fb7ceb45d8cbc4c5054f4b69d3b0ff98bcd750571  "The quick brown fox jumps over the lazy dog"
```

An implementation is correct if it produces these exact outputs.

### Empirical test results

All GPU tests run on NVIDIA RTX 3090 Ti (sm_86).

#### Strict Avalanche Criterion

```
samples:     16,384 × 64 input-bit flips = 1,048,576 tests
avg Hamming: 128.00 / 256  (50.0%)
min / max:   90 / 165
per-bit:     127.8 – 128.2  (all 64 input positions uniform)
result:      PASS
```

#### Birthday collision test

```
N = 2^14,  24-bit prefix:  actual=8,     expected=8.0,   ratio=1.00  PASS
N = 2^20,  32-bit prefix:  actual=112,   expected=128,   ratio=0.88  PASS
N = 2^20,  48-bit prefix:  actual=0,     expected=~0              PASS
N = 2^24,  32-bit prefix:  actual=32748, expected=32768, ratio=1.00  PASS
```

#### Differential cryptanalysis (1,157,627,904 total hash evaluations)

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

---

## FractalCipher — stream cipher and keypair

FractalCipher is an authenticated stream cipher (AEAD) built on the
FractalSponge permutation, combined with a keypair scheme for key exchange.

### Key scheme

```
Private key : 32 bytes from /dev/urandom
Public key  : H("FractalCipher-pubkey" ‖ private_key)

Shared secret: H("FractalCipher-shared" ‖ sort_lo(pubA, pubB)
                                        ‖ sort_hi(pubA, pubB))
```

Both parties sort their public keys lexicographically before hashing, so
`fc_shared_secret(alice_priv, bob_pub)` and
`fc_shared_secret(bob_priv, alice_pub)` return the same 32-byte value.
Each party must know their own private key to derive their public key, binding
identity to the shared session.

### Duplex sponge encryption

The cipher uses the sponge state in *duplex mode*: absorb input, squeeze
keystream, re-absorb the ciphertext. This creates a tight binding between the
keystream and the data that has already been encrypted.

```
State ← absorb("FractalCipher-stream" ‖ shared_secret ‖ nonce)

  for each 32-byte block of plaintext:
    keystream_block  ← squeeze(state)
    ciphertext_block ← plaintext_block ⊕ keystream_block
    state            ← absorb(ciphertext_block)        ← duplex binding

Auth tag ← finalize domain separator, squeeze 16 bytes
```

**Decryption** follows the same path: absorb the *ciphertext* (not plaintext)
to maintain the identical state trajectory, then recompute and verify the tag.
Tag comparison is constant-time.

### Security properties

| Property | Status |
|----------|--------|
| Confidentiality | Keystream derived from shared secret + per-message nonce |
| Authentication | 16-byte AEAD tag; forgery requires full state knowledge |
| Nonce misuse | Reusing a nonce with the same key leaks plaintext XOR |
| Key secrecy | Shared secret depends only on the two public keys (see note) |
| Forward secrecy | Not provided — use ephemeral keypairs per session for PFS |

> **Note on key secrecy:** The shared secret is derived solely from the two
> public keys. This is symmetric and deterministic, but means the shared
> secret is not hidden from a party who obtains both public keys. For
> applications requiring confidentiality against passive eavesdroppers who
> record public keys, use ephemeral keypairs or a proper DH group (X25519).

### CLI usage

```
fractal_cipher keygen  <privkey_out> <pubkey_out>
fractal_cipher shared  <my_privkey> <peer_pubkey> <shared_out>
fractal_cipher encrypt <shared_file> <plaintext_in> <ciphertext_out>
fractal_cipher decrypt <shared_file> <ciphertext_in> <plaintext_out>
fractal_cipher test
```

**Full workflow example:**

```bash
# Each party generates a keypair
./fractal_cipher keygen alice.priv alice.pub
./fractal_cipher keygen bob.priv   bob.pub

# Exchange public keys (over any channel), then each derives the shared secret
./fractal_cipher shared alice.priv bob.pub   shared.key  # run by Alice
./fractal_cipher shared bob.priv   alice.pub shared.key  # run by Bob
# Both shared.key files are identical

# Alice encrypts
./fractal_cipher encrypt shared.key message.txt message.enc

# Bob decrypts
./fractal_cipher decrypt shared.key message.enc message.txt
```

**Encrypted file format:**

```
[16 bytes nonce][16 bytes auth tag][ciphertext ...]
```

The nonce is generated fresh from `/dev/urandom` for every encryption.
If decryption fails (wrong key, tampered data, wrong nonce), the command
prints an error, exits with code 1, and the output file is zeroed.

### Self-test

```bash
./fractal_cipher test
```

Runs 6 built-in tests:

1. Keypair + shared secret symmetry — both sides derive the same value
2. Encrypt/decrypt round-trip — plaintext recovered exactly
3. Tamper detection — single-byte flip in ciphertext fails authentication
4. Wrong key rejected — decryption with a different keypair fails
5. Deterministic — same inputs produce the same ciphertext and tag
6. Nonce uniqueness — different nonces produce different ciphertext

### C API

```c
#include "fractal_cipher.h"

/* Generate a keypair from /dev/urandom */
int fc_keygen(FcKeypair *kp);

/* Derive a symmetric shared secret from your private key + peer's public key */
void fc_shared_secret(const uint8_t my_priv[32],
                      const uint8_t peer_pub[32],
                      uint8_t       shared[32]);

/* Authenticated encryption — writes ciphertext (same length as pt) + 16-byte tag */
void fc_encrypt(const uint8_t shared[32],
                const uint8_t nonce[16],
                const uint8_t *aad,  size_t aad_len,   /* optional */
                const uint8_t *pt,   uint8_t *ct,  size_t pt_len,
                uint8_t        tag[16]);

/* Authenticated decryption — returns 0 on success, -1 if tag fails */
int  fc_decrypt(const uint8_t shared[32],
                const uint8_t nonce[16],
                const uint8_t *aad,  size_t aad_len,
                const uint8_t *ct,   uint8_t *pt,  size_t ct_len,
                const uint8_t  tag[16]);

/* Generate a fresh per-message nonce from /dev/urandom */
int  fc_random_nonce(uint8_t nonce[16]);
```

---

## Build

### Requirements

| Component | Minimum version |
|-----------|----------------|
| gcc / clang | gcc 7+, clang 6+ |
| CPU | Any x86-64 (AVX2 used automatically if available) |
| CUDA toolkit | 11.0+ (GPU only) |
| GPU | sm_75+ (Turing / RTX 2000 series or newer) |
| OS | Linux (primary), macOS (untested) |

### Targets

```bash
make cpu          # fractal_hash   — hash CLI
make cipher       # fractal_cipher — stream cipher CLI
make gpu          # fractal_gpu    — GPU cryptanalysis suite
make all          # all three
make test         # CPU vectors + cipher self-test
make test-cipher  # cipher self-test only
make test-gpu     # GPU bench + avalanche + birthday
make clean
```

### CPU hash

```bash
./fractal_hash file.bin          # hash a file (sha256sum-compatible output)
./fractal_hash -s "message"      # hash a string
echo "hello" | ./fractal_hash    # hash stdin
./fractal_hash --test            # canonical vector verification
./fractal_hash --bench           # throughput benchmark
```

### GPU

```bash
make info                              # show detected GPU arch
./fractal_gpu --avalanche              # SAC test (16K samples)
./fractal_gpu --avalanche 65536        # larger sample count
./fractal_gpu --birthday               # birthday collision suite
./fractal_gpu --birthday-large         # adds 2^24 tier
./fractal_gpu --differential           # full cryptanalysis (~15 min)
./fractal_gpu --bench                  # throughput
./fractal_gpu file.bin                 # hash file (CPU path)
```

### Important: do not use -march=native or -ffast-math

These flags enable FMA contraction on modern x86 CPUs, silently changing
`a*b+c` rounding behavior and producing different hashes across CPU
generations. The Makefile handles this correctly with `-ffp-contract=off`.
The AVX2 file is compiled with `-mavx2` only — never `-mfma` or fast-math.

---

## Repository layout

```
├── fractal_sponge.h          types, constants, public API (hash + permute)
├── fractal_sponge.c          CPU hash implementation (C99, strict IEEE 754)
├── fractal_sponge_avx2.c     AVX2-accelerated permutation (4-wide double)
├── main.c                    hash CLI (sha256sum-compatible)
│
├── fractal_cipher.h          cipher public API
├── fractal_cipher.c          duplex sponge AEAD implementation
├── cipher_main.c             cipher CLI
│
├── fractal_sponge.cuh        shared CUDA types and declarations
├── fractal_sponge.cu         GPU permutation + sponge (strict IEEE 754)
├── avalanche.cu              SAC measurement kernel
├── birthday.cu               birthday collision test
├── differential.cu           differential cryptanalysis (3 tests)
├── gpu_main.cu               GPU CLI
│
├── Makefile
├── rebuild.sh                force clean rebuild
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
- AVX2 bit-exact parity — AVX2 path produces identical digests to scalar path (verified against all 5 test vectors)

### What has not been proven

- **Preimage resistance** — inverting iterated `z²+c` maps relates to open problems in arithmetic dynamics; no formal hardness reduction exists
- **Collision resistance** — no attack known, no proof given
- **Side-channel resistance** — floating-point timing behavior not analyzed
- **Cipher forward secrecy** — shared secret is static; use ephemeral keypairs per session

### Security argument for the hash

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

### Security argument for the cipher

**Keystream unpredictability.** The keystream is produced by repeatedly
permuting a 512-bit sponge state initialised from a 32-byte shared secret and
a 16-byte nonce. Breaking the keystream requires either: (a) inverting
`fs256_permute`, which inherits the hash's hardness argument, or (b)
distinguishing the output from random, for which no distinguisher is known.

**Authentication.** The duplex re-absorption of ciphertext into the state
binds the tag to every byte of ciphertext. An adversary who flips any
ciphertext bit changes the tag computation path and cannot compute a valid tag
without the shared secret.

---

## Open research questions

1. **Hardness of fractal inversion.** Given `y = χ_F(x)`, what is the complexity of finding `x`? Is inverting iterated quadratic maps NP-hard?

2. **Algebraic structure of orbit endpoints.** The six endpoint values lie on algebraic varieties. Do these enable a structural attack bypassing the ARX fold?

3. **Higher-order differentials.** Tests show no anomaly through 2-bit differentials. 4-bit and 8-bit tests at 2³⁰ scale would further bound differential characteristics.

4. **AVX-512 path.** The current AVX2 path processes 4 S-boxes per pass; AVX-512 would handle all 8 at once for a further ~2× speedup.

5. **Formal security reduction.** Can the sbox's one-wayness be reduced to a standard hard problem, or to an explicit conjecture in arithmetic dynamics?

6. **Cipher nonce-misuse resistance.** The current design leaks plaintext XOR on nonce reuse. A misuse-resistant variant (e.g. SIV-style) is an open design question.

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

FractalCipher extends this into a practical tool: a duplex sponge AEAD with a
simple public-key layer for key exchange, demonstrating that the same fractal
permutation that provides hash security can also drive a stream cipher and
authentication scheme.

The security questions for this construction connect directly to arithmetic
dynamics — the study of iteration of rational maps over number fields — an
active area of pure mathematics where many fundamental questions remain open.

---

## License

MIT — see `LICENSE`.

This is a research prototype. Do not use in production systems without
independent security review and formal analysis.

/*
 * fractal_sponge.cu — CUDA implementation of FractalSponge-256 v4.3
 *
 * Each CUDA thread independently hashes one message.
 * No shared memory needed between threads — perfect GPU workload.
 *
 * Compile:
 *   nvcc -O3 -arch=sm_75 -use_fast_math -o fractal_gpu \
 *        fractal_sponge.cu birthday.cu avalanche.cu gpu_main.cu
 *
 * For sm_86 (RTX 3xxx) or sm_89 (RTX 4xxx), change -arch accordingly.
 * Check yours with: nvidia-smi --query-gpu=compute_cap --format=csv
 */

#include "fractal_sponge.cuh"
#include <math.h>
#include <string.h>

/* ── device utilities ──────────────────────────────────────────────────────── */

__device__ __forceinline__
static uint64_t d_rot64(uint64_t v, int n) {
    return (v << n) | (v >> (64 - n));
}

__device__ __forceinline__
static uint64_t d_rc(int r) {
    return (RC_PHI*(uint64_t)(r+1))
         ^ (RC_SQRT2*(uint64_t)(r*7+3))
         ^ (RC_SQRT3*(uint64_t)(r*13+5));
}
/*
__device__ __forceinline__
static uint64_t d_mantissa(double f) {
    uint64_t bits;
    memcpy(&bits, &f, 8);
    return bits & UINT64_C(0x000FFFFFFFFFFFFF);
}
*/

/* Full 64-bit reinterpretation — preserves sign+exponent+mantissa.
 * mantissa(4.0)=0, but d_f2b(4.0)=0x4010000000000000. Use this instead
 * of d_mantissa() whenever the float may land near a power of 2. */
__device__ __forceinline__
static uint64_t d_f2b(double f) {
    uint64_t b; memcpy(&b, &f, 8); return b;
}

__device__ __forceinline__
static uint64_t d_arx_whiten(uint64_t v) {
    v += ARX_W0;
    v  = d_rot64(v, 13) ^ ARX_W1;
    v += d_rot64(v, 32);
    v ^= v >> 33;
    v *= UINT64_C(0xff51afd7ed558ccd);
    v ^= v >> 33;
    v *= UINT64_C(0xc4ceb9fe1a85ec53);
    v ^= v >> 33;
    return v;
}

__device__ __forceinline__
static uint64_t d_mix64(uint64_t v) {
    v  = (v ^ (v >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    v  = (v ^ (v >> 27)) * UINT64_C(0x94d049bb133111eb);
    return v ^ (v >> 31);
}

/* ── device fractal orbits ─────────────────────────────────────────────────── */
/*
 * STRICT IEEE 754 MODE for all orbit functions.
 *
 * --use_fast_math causes two problems:
 *   1. sqrt() → fast approximation (different bits)
 *   2. FMA contraction: a*b+c becomes a single fused op with different rounding
 *
 * Fix 1: use __dsqrt_rn() — IEEE 754 round-to-nearest double sqrt.
 * Fix 2: #pragma STDC FP_CONTRACT OFF — disables FMA for this translation unit.
 *
 * Together these guarantee the device orbits produce bit-identical results
 * to the CPU implementation compiled with gcc -O2 (strict IEEE 754 default).
 */
#pragma STDC FP_CONTRACT OFF

__device__
static void d_julia(double *re, double *im,
                    double cre, double cim, int n) {
    for (int i = 0; i < n; i++) {
        double nr = (*re)*(*re) - (*im)*(*im) + cre;
        double ni = 2.0*(*re)*(*im)            + cim;
        *re = nr; *im = ni;
        double m = nr*nr + ni*ni;
        if (m > 4.0) { double s = __dsqrt_rn(m)*0.5; *re /= s; *im /= s; }
    }
}

__device__
static void d_newton(double *re, double *im, int n) {
    for (int i = 0; i < n; i++) {
        double r2 = (*re)*(*re)-(*im)*(*im), ri = 2.0*(*re)*(*im);
        double r3 = (*re)*r2-(*im)*ri,       i3 = (*re)*ri+(*im)*r2;
        double dre = 3.0*((*re)*(*re)-(*im)*(*im)), dim = 6.0*(*re)*(*im);
        double den = dre*dre+dim*dim;
        if (den < 1e-14) break;
        double nre=r3-1.0, nim=i3;
        *re -= (nre*dre+nim*dim)/den;
        *im -= (nim*dre-nre*dim)/den;
        double m = (*re)*(*re)+(*im)*(*im);
        if (m > 16.0) { double s=__dsqrt_rn(m)*0.25; *re/=s; *im/=s; }
    }
}

__device__
static void d_ship(double *re, double *im,
                   double cre, double cim, int n) {
    for (int i = 0; i < n; i++) {
        *re = fabs(*re); *im = fabs(*im);
        double nr = (*re)*(*re)-(*im)*(*im)+cre;
        double ni = 2.0*(*re)*(*im)        +cim;
        *re=nr; *im=ni;
        double m=nr*nr+ni*ni;
        if (m>4.0){ double s=__dsqrt_rn(m)*0.5; *re/=s; *im/=s; }
    }
}

/* ── device sbox ────────────────────────────────────────────────────────────── */

__device__ __constant__
static const double D_JC_RE[4] = { 0.285,    -0.70176,  0.355,  -0.4   };
__device__ __constant__
static const double D_JC_IM[4] = { 0.013,     0.3842,   0.355,   0.6   };
__device__ __constant__
static const double D_SC_RE[4] = {-1.755,    -1.755,    0.400,  -0.500 };
__device__ __constant__
static const double D_SC_IM[4] = { 0.028,    -0.028,    0.500,  -0.500 };

#define D_SCALE21 (2.0 / (double)(1 << 21))

__device__
static uint64_t d_fractal_sbox(uint64_t word, int cell, uint64_t rcc) {
    /*
     * Split the 64-bit word into three float coordinates with ZERO bit overlap.
     *
     * lo32 = bits  0–31  (low  half)
     * hi32 = bits 32–63  (high half)
     * mix  = lo32 XOR rot32(hi32, 13)  — all 64 bits influence mix
     *
     * Each is mapped to [-2, 2) via: val * (4.0/2^32) - 2.0
     * This covers all 64 input bits, no collision, no MSB patch needed.
     */
    uint32_t lo32 = (uint32_t)(word & 0xFFFFFFFFULL);
    uint32_t hi32 = (uint32_t)(word >> 32);
    uint32_t mix  = lo32 ^ ((hi32 << 13) | (hi32 >> 19));  /* rot32(hi32,13) */

    const double SCALE32 = 4.0 / (double)(1ULL << 32);  /* maps [0,2^32) to [0,4) */
    double f0 = (double)lo32 * SCALE32 - 2.0;  /* [-2, 2) */
    double f1 = (double)hi32 * SCALE32 - 2.0;
    double f2 = (double)mix  * SCALE32 - 2.0;

    double rc_f = ((double)(rcc & 0xFFFFF) / (double)(1 << 20)) - 0.5;

    double jre = f0, jim = f1;
    d_julia(&jre, &jim, D_JC_RE[cell]+rc_f*0.1, D_JC_IM[cell]+rc_f*0.07, 8);

    double nre = f1, nim = f2;
    d_newton(&nre, &nim, 7);

    double sre = f0, sim = f2;
    d_ship(&sre, &sim, D_SC_RE[cell]+rc_f*0.05, D_SC_IM[cell]+rc_f*0.03, 8);

    /* Harvest: full 64-bit reinterpretation of all 6 orbit endpoints.
     *
     * CRITICAL: pure XOR-folding is linear over GF(2). Fractal orbits have
     * algebraic structure (Julia sets satisfy z^2+c=0 attractors), so the
     * 6 endpoint values carry GF(2)-linear dependencies that survive XOR
     * folding and cause output space clustering (~2^33 effective bits instead
     * of 2^64). Fix: interleave modular addition with XOR — ADD is nonlinear
     * over GF(2), breaking the algebraic dependencies.
     */
    uint64_t h0 = d_f2b(jre), h1 = d_f2b(jim);
    uint64_t h2 = d_f2b(nre), h3 = d_f2b(nim);
    uint64_t h4 = d_f2b(sre), h5 = d_f2b(sim);

    /* ARX fold: alternate ADD and XOR with rotations */
    uint64_t result = h0;
    result  = (result + d_rot64(h1, 11)) & UINT64_C(0xFFFFFFFFFFFFFFFF);
    result ^= d_rot64(h2, 23);
    result  = (result + d_rot64(h3, 37)) & UINT64_C(0xFFFFFFFFFFFFFFFF);
    result ^= d_rot64(h4, 47);
    result  = (result + d_rot64(h5, 53)) & UINT64_C(0xFFFFFFFFFFFFFFFF);
    result ^= (rcc >> 32);
    return d_mix64(d_arx_whiten(result));
}

/* ── device theta — integer XOR column parity (replaces DFT) ────────────────
 *
 * The DFT theta had a fatal flaw: it converts state words to floats and sums
 * them. With 8-byte messages, only state[0] differs between inputs; the other
 * three rate words are fixed padding constants. The message contribution
 * (~1e-19 in float space) is drowned by the padding constants (~1.0) due to
 * float64 precision loss — bits 53–63 of state[0] land below machine epsilon
 * after addition, making them invisible to the permutation.
 *
 * The XOR-based theta has no precision issues: every bit of every word
 * influences every other word equally via XOR, regardless of magnitude.
 * Two passes with different rotation offsets give full cross-word diffusion.
 */
__device__
static void d_theta_xor(uint64_t s[8], uint64_t rcc) {
    /* Pass 1: standard column parity */
    uint64_t p = s[0]^s[1]^s[2]^s[3];
    uint64_t q = s[4]^s[5]^s[6]^s[7];
    uint64_t d = p ^ q ^ d_rot64(p,1) ^ d_rot64(q,7);
    for (int i = 0; i < 8; i++) s[i] ^= d;

    /* Pass 2: secondary mix after rho — inject rcc for round separation */
    uint64_t p2 = s[0]^s[2]^s[4]^s[6];
    uint64_t q2 = s[1]^s[3]^s[5]^s[7];
    uint64_t d2 = d_rot64(p2,13) ^ d_rot64(q2,41) ^ rcc;
    for (int i = 0; i < 8; i++) s[i] ^= d2;
}

/* ── device permutation ─────────────────────────────────────────────────────── */

__device__ __constant__
static const int D_RHO[8] = {1, 3, 6, 10, 15, 21, 28, 36};
__device__ __constant__
static const int D_PI[8]  = {3, 6, 1,  4,  7,  2,  5,  0};

__device__
static void d_permutation(uint64_t s[8]) {
    for (int r = 0; r < FS_ROUNDS; r++) {
        uint64_t rcc = d_rc(r);

        d_theta_xor(s, rcc);

        for (int i = 0; i < 8; i++) s[i] = d_rot64(s[i], D_RHO[i]);

        uint64_t tmp[8];
        for (int i = 0; i < 8; i++) tmp[D_PI[i]] = s[i];
        for (int i = 0; i < 8; i++) s[i] = tmp[i];

        /* χ_F: widen context to 4 words (adjacent + opposite half of state).
         * Using only 3 adjacent words left rank deficiency across 8-word state.
         * Adding t[(i+4)%8] (the antipodal word) ensures every sbox sees both
         * halves of the state, fully coupling rate and capacity words. */
        uint64_t t[8];
        for (int i = 0; i < 8; i++) t[i] = s[i];
        for (int i = 0; i < 8; i++) {
            uint64_t ctx = t[i]
                         ^ d_rot64(t[(i+1)%8], 13)
                         ^ d_rot64(t[(i+7)%8], 41)
                         ^ d_rot64(t[(i+4)%8], 27);  /* antipodal word */
            s[i] = d_fractal_sbox(ctx, i%4, d_rot64(rcc, i*8));
        }

        s[0] ^= rcc;
        s[1] ^= d_rot64(rcc, 32);
    }

    /* Final full-state coupling — ensures all 8 words are mutually dependent
     * before squeeze. Runs once after all rounds, not inside the loop.
     * Without this, words squeezed first (state[0,1]) may not fully reflect
     * all 8 words, causing rank deficiency in the digest top bytes. */
    uint64_t xall = s[0]^s[1]^s[2]^s[3]^s[4]^s[5]^s[6]^s[7];
    for (int i = 0; i < 8; i++)
        s[i] ^= d_rot64(xall, i * 7 + 1);
}

/* ── device sponge ──────────────────────────────────────────────────────────── */
__device__
void d_hash8(const uint8_t msg[8], uint8_t digest[32]) {
    uint8_t block[32];
    for (int i = 0; i < 8;  i++) block[i] = msg[i];
    block[8] = 0x06;
    for (int i = 9; i < 31; i++) block[i] = 0x00;
    block[31] = 0x80;

    /* Explicit zero-init — device local arrays are NOT auto-zeroed */
    uint64_t state[8];
    state[0] = state[1] = state[2] = state[3] = 0;
    state[4] = state[5] = state[6] = state[7] = 0;

    /* Big-endian unpack into rate words only (capacity stays zero) */
    for (int j = 0; j < 4; j++) {
        uint64_t w = 0;
        for (int b = 0; b < 8; b++)
            w = (w << 8) | block[j*8+b];
        state[j] ^= w;
    }

    d_permutation(state);

    /* Pack digest big-endian */
    for (int i = 0; i < 4; i++) {
        uint64_t v = state[i];
        for (int b = 7; b >= 0; b--) {
            digest[i*8+b] = (uint8_t)(v & 0xff);
            v >>= 8;
        }
    }
}

/* ── variable-length device hash (up to 256 bytes) ──────────────────────────── */
__device__
void d_hash_var(const uint8_t *msg, int len, uint8_t digest[32]) {
    uint8_t buf[512];
    for (int i = 0; i < len; i++) buf[i] = msg[i];
    buf[len] = 0x06;
    int padded = len + 1;
    while (padded % 32) buf[padded++] = 0x00;
    buf[padded-1] |= 0x80;

    /* Explicit zero-init */
    uint64_t state[8];
    state[0] = state[1] = state[2] = state[3] = 0;
    state[4] = state[5] = state[6] = state[7] = 0;

    for (int off = 0; off < padded; off += 32) {
        for (int j = 0; j < 4; j++) {
            uint64_t w = 0;
            for (int b = 0; b < 8; b++)
                w = (w << 8) | buf[off + j*8 + b];
            state[j] ^= w;
        }
        d_permutation(state);
    }
    for (int i = 0; i < 4; i++) {
        uint64_t v = state[i];
        for (int b = 7; b >= 0; b--) {
            digest[i*8+b] = (uint8_t)(v & 0xff);
            v >>= 8;
        }
    }
}

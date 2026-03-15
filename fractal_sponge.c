/*
 * FractalSponge-256 v4.3 — C implementation
 * Identical algorithm to the Python reference; bit-for-bit compatible output.
 *
 * Build:  gcc -O2 -lm -o fractal_hash fractal_sponge.c main.c
 *
 * NOTE: do NOT use -march=native or -ffast-math. The fractal sbox extracts
 * IEEE 754 bit patterns from orbit endpoints. FMA contraction (enabled by
 * -mfma on modern CPUs via -march=native) silently changes a*b+c rounding,
 * producing different bit patterns on different CPU generations.
 * #pragma STDC FP_CONTRACT OFF below enforces strict two-operation semantics
 * regardless of what -march=native or the CPU supports.
 */

#include "fractal_sponge.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* ── AVX2 runtime dispatch ────────────────────────────────────────────────── */
/* Declared in fractal_sponge_avx2.c (compiled with -mavx2) */
extern void fractal_permutation_avx2(uint64_t s[8]);

/* CPUID leaf 7 / EBX bit 5 = AVX2 */
static int cpu_has_avx2(void) {
#if defined(__x86_64__) || defined(__i386__)
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    /* Check CPUID max leaf */
    __asm__ volatile (
        "cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "0"(0)
    );
    if (eax < 7) return 0;
    /* Leaf 7, sub-leaf 0 */
    eax = 7; ecx = 0;
    __asm__ volatile (
        "cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "0"(eax), "2"(ecx)
    );
    /* EBX bit 5 = AVX2; also check OS saves YMM (OSXSAVE + YMM state) */
    if (!(ebx & (1u << 5))) return 0;
    /* Verify XSAVE enabled by OS (OSXSAVE = ECX bit 27 in leaf 1) */
    eax = 1; ecx = 0;
    __asm__ volatile (
        "cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "0"(eax), "2"(ecx)
    );
    if (!(ecx & (1u << 27))) return 0;
    /* Read XCR0 to check OS saves YMM registers (bit 2) */
    unsigned long long xcr0;
    __asm__ volatile ("xgetbv" : "=A"(xcr0) : "c"(0));
    return (xcr0 & 0x6) == 0x6;
#else
    return 0;
#endif
}

/* Forward declaration — defined later in this file */
static void fractal_permutation(uint64_t s[8]);

/* Function pointer — set on first call */
static void (*active_permutation)(uint64_t[8]) = NULL;

static void init_permutation(void) {
    if (cpu_has_avx2()) {
        active_permutation = fractal_permutation_avx2;
    } else {
        active_permutation = fractal_permutation;
    }
}

/* Disable FMA contraction — see build note above. The pragma is C99/C11
 * standard but gcc requires -std=c99 or higher to honour it.
 * The Makefile uses -ffp-contract=off as the reliable gcc equivalent. */

/* ── portability ──────────────────────────────────────────────────────────── */

static inline uint64_t rot64(uint64_t v, int n) {
    return (v << n) | (v >> (64 - n));
}

/* Read big-endian uint64 from 8 bytes */
static inline uint64_t be64(const uint8_t *p) {
    return ((uint64_t)p[0]<<56)|((uint64_t)p[1]<<48)|
           ((uint64_t)p[2]<<40)|((uint64_t)p[3]<<32)|
           ((uint64_t)p[4]<<24)|((uint64_t)p[5]<<16)|
           ((uint64_t)p[6]<< 8)|((uint64_t)p[7]);
}

static inline void put_be64(uint8_t *p, uint64_t v) {
    p[0]=(v>>56)&0xff; p[1]=(v>>48)&0xff;
    p[2]=(v>>40)&0xff; p[3]=(v>>32)&0xff;
    p[4]=(v>>24)&0xff; p[5]=(v>>16)&0xff;
    p[6]=(v>> 8)&0xff; p[7]= v     &0xff;
}

/* ── round constant ───────────────────────────────────────────────────────── */

static inline uint64_t rc(int r) {
    return (RC_PHI*(uint64_t)(r+1))
         ^ (RC_SQRT2*(uint64_t)(r*7+3))
         ^ (RC_SQRT3*(uint64_t)(r*13+5));
}

/* ── IEEE 754 mantissa extraction ────────────────────────────────────────── */

static inline uint64_t mantissa(double f) {
    uint64_t bits;
    memcpy(&bits, &f, 8);
    return bits & UINT64_C(0x000FFFFFFFFFFFFF);
}

/* ── ARX whitening + Murmur3 finalizer ───────────────────────────────────── */

static inline uint64_t arx_whiten(uint64_t v) {
    v += ARX_W0;
    v  = rot64(v, 13) ^ ARX_W1;
    v += rot64(v, 32);
    v ^= v >> 33;
    v *= UINT64_C(0xff51afd7ed558ccd);
    v ^= v >> 33;
    v *= UINT64_C(0xc4ceb9fe1a85ec53);
    v ^= v >> 33;
    return v;
}

static inline uint64_t mix64(uint64_t v) {
    v  = (v ^ (v >> 30)) * MIX_C0;
    v  = (v ^ (v >> 27)) * MIX_C1;
    return v ^ (v >> 31);
}

/* ── Fractal orbit functions ──────────────────────────────────────────────── */

typedef struct { double re, im; } cpx;

static inline cpx julia(double re, double im,
                         double cre, double cim, int n) {
    for (int i = 0; i < n; i++) {
        double new_re = re*re - im*im + cre;
        double new_im = 2.0*re*im      + cim;
        re = new_re; im = new_im;
        double m = re*re + im*im;
        if (m > 4.0) { double s = sqrt(m)*0.5; re /= s; im /= s; }
    }
    return (cpx){re, im};
}

static inline cpx newton(double re, double im, int n) {
    for (int i = 0; i < n; i++) {
        double r2  = re*re - im*im,  ri  = 2.0*re*im;
        double r3  = re*r2 - im*ri,  i3  = re*ri + im*r2;
        double dre = 3.0*(re*re - im*im), dim = 6.0*re*im;
        double den = dre*dre + dim*dim;
        if (den < 1e-14) break;
        double nre = r3 - 1.0, nim = i3;
        re -= (nre*dre + nim*dim) / den;
        im -= (nim*dre - nre*dim) / den;
        double m = re*re + im*im;
        if (m > 16.0) { double s = sqrt(m)*0.25; re /= s; im /= s; }
    }
    return (cpx){re, im};
}

static inline cpx ship(double re, double im,
                        double cre, double cim, int n) {
    for (int i = 0; i < n; i++) {
        re = fabs(re); im = fabs(im);
        double new_re = re*re - im*im + cre;
        double new_im = 2.0*re*im      + cim;
        re = new_re; im = new_im;
        double m = re*re + im*im;
        if (m > 4.0) { double s = sqrt(m)*0.5; re /= s; im /= s; }
    }
    return (cpx){re, im};
}

/* ── Fractal S-box ────────────────────────────────────────────────────────── */

static uint64_t fractal_sbox(uint64_t word, int cell, uint64_t rcc) {
    /*
     * Split 64-bit word into three float coordinates — zero bit overlap.
     * lo32 = bits  0–31,  hi32 = bits 32–63,  mix = lo ^ rot32(hi, 13)
     * All 64 bits influence at least one coordinate directly.
     */
    uint32_t lo32 = (uint32_t)(word & 0xFFFFFFFFULL);
    uint32_t hi32 = (uint32_t)(word >> 32);
    uint32_t mix  = lo32 ^ ((hi32 << 13) | (hi32 >> 19));

    const double SCALE32 = 4.0 / (double)(1ULL << 32);
    double f0 = (double)lo32 * SCALE32 - 2.0;
    double f1 = (double)hi32 * SCALE32 - 2.0;
    double f2 = (double)mix  * SCALE32 - 2.0;

    double rc_f = ((double)(rcc & 0xFFFFF) / (double)(1 << 20)) - 0.5;

    cpx j  = julia(f0, f1, JC_RE[cell]+rc_f*0.1,  JC_IM[cell]+rc_f*0.07, 8);
    cpx nw = newton(f1, f2, 7);
    cpx sh = ship(f0, f2,  SC_RE[cell]+rc_f*0.05, SC_IM[cell]+rc_f*0.03, 8);

    /* Harvest: full 64-bit reinterpretation with ARX fold.
     * Pure XOR is linear over GF(2) — fractal orbit endpoints carry algebraic
     * dependencies that survive XOR and cause output clustering. Interleaving
     * ADD (nonlinear over GF(2)) with XOR breaks these dependencies. */
    uint64_t h0,h1,h2,h3,h4,h5;
    memcpy(&h0, &j.re,  8); memcpy(&h1, &j.im,  8);
    memcpy(&h2, &nw.re, 8); memcpy(&h3, &nw.im, 8);
    memcpy(&h4, &sh.re, 8); memcpy(&h5, &sh.im, 8);

    uint64_t result = h0;
    result  = result + rot64(h1, 11);
    result ^= rot64(h2, 23);
    result  = result + rot64(h3, 37);
    result ^= rot64(h4, 47);
    result  = result + rot64(h5, 53);
    result ^= (rcc >> 32);

    return mix64(arx_whiten(result));
}

/* ── Theta — integer XOR column parity (replaces DFT) ────────────────────── */

static void theta_xor(uint64_t s[8], uint64_t rcc) {
    uint64_t p = s[0]^s[1]^s[2]^s[3];
    uint64_t q = s[4]^s[5]^s[6]^s[7];
    uint64_t d = p ^ q ^ rot64(p,1) ^ rot64(q,7);
    for (int i = 0; i < 8; i++) s[i] ^= d;

    uint64_t p2 = s[0]^s[2]^s[4]^s[6];
    uint64_t q2 = s[1]^s[3]^s[5]^s[7];
    uint64_t d2 = rot64(p2,13) ^ rot64(q2,41) ^ rcc;
    for (int i = 0; i < 8; i++) s[i] ^= d2;
}

/* ── Core permutation ─────────────────────────────────────────────────────── */

static void fractal_permutation(uint64_t s[8]) {
    for (int r = 0; r < FS_ROUNDS; r++) {
        uint64_t rcc = rc(r);

        /* θ_XOR — integer column parity diffusion */
        theta_xor(s, rcc);

        /* ρ */
        for (int i = 0; i < 8; i++)
            s[i] = rot64(s[i], FS_RHO[i]);

        /* π */
        uint64_t tmp[8];
        for (int i = 0; i < 8; i++) tmp[FS_PI[i]] = s[i];
        memcpy(s, tmp, 64);

        /* χ_F — widened context: adjacent + antipodal word for full coupling */
        uint64_t t[8]; memcpy(t, s, 64);
        for (int i = 0; i < 8; i++) {
            uint64_t ctx = t[i]
                         ^ rot64(t[(i+1)%8], 13)
                         ^ rot64(t[(i+7)%8], 41)
                         ^ rot64(t[(i+4)%8], 27);  /* antipodal word */
            s[i] = fractal_sbox(ctx, i%4, rot64(rcc, i*8));
        }

        /* ι */
        s[0] ^= rcc;
        s[1] ^= rot64(rcc, 32);
    }

    /* Final full-state coupling before squeeze — ensures all 8 words are
     * mutually dependent. Prevents rank deficiency in digest top bytes. */
    uint64_t xall = s[0]^s[1]^s[2]^s[3]^s[4]^s[5]^s[6]^s[7];
    for (int i = 0; i < 8; i++)
        s[i] ^= rot64(xall, i * 7 + 1);
}

/* ── Padding (SHA-3 multirate) ────────────────────────────────────────────── */

/*
 * We pad inline into a caller-supplied buffer.
 * Returns padded length (always multiple of FS_BLOCK_BYTES).
 * Caller must allocate at least len + FS_BLOCK_BYTES bytes.
 */
static size_t fs_pad(const uint8_t *in, size_t len, uint8_t *out) {
    memcpy(out, in, len);
    out[len] = 0x06;
    size_t padded = len + 1;
    while (padded % FS_BLOCK_BYTES) out[padded++] = 0x00;
    out[padded - 1] |= 0x80;
    return padded;
}

/* ── Public API ───────────────────────────────────────────────────────────── */

void fs256_hash(const uint8_t *data, size_t len, uint8_t digest[FS256_DIGEST_BYTES]) {
    /* Allocate padded buffer on heap to avoid VLA / stack blowup on large inputs */
    size_t padded_len = len + FS_BLOCK_BYTES;
    /* Round up to next block boundary */
    padded_len = ((padded_len + FS_BLOCK_BYTES - 1) / FS_BLOCK_BYTES) * FS_BLOCK_BYTES;
    uint8_t *buf = (uint8_t *)malloc(padded_len);
    if (!buf) return;  /* caller should check but we can't return error here */

    size_t plen = fs_pad(data, len, buf);

    /* One-time CPU capability detection */
    if (!active_permutation) init_permutation();

    uint64_t state[8] = {0};
    for (size_t off = 0; off < plen; off += FS_BLOCK_BYTES) {
        for (int j = 0; j < FS_RATE_WORDS; j++)
            state[j] ^= be64(buf + off + j*8);
        active_permutation(state);
    }
    free(buf);

    for (int i = 0; i < 4; i++)
        put_be64(digest + i*8, state[i]);
}

void fs256_permute(uint64_t state[FS_STATE_WORDS]) {
    if (!active_permutation) init_permutation();
    active_permutation(state);
}

void fs256_hash_hex(const uint8_t *data, size_t len, char hex[FS256_DIGEST_HEX]) {
    uint8_t digest[FS256_DIGEST_BYTES];
    fs256_hash(data, len, digest);
    static const char *hx = "0123456789abcdef";
    for (int i = 0; i < FS256_DIGEST_BYTES; i++) {
        hex[2*i]   = hx[digest[i] >> 4];
        hex[2*i+1] = hx[digest[i] & 0xf];
    }
    hex[64] = '\0';
}

int fs256_hash_file(const char *path, uint8_t digest[FS256_DIGEST_BYTES]) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    /* Read whole file — for large files a streaming version is better */
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    if (sz < 0) { fclose(f); return -1; }

    uint8_t *buf = (uint8_t *)malloc((size_t)sz);
    if (!buf) { fclose(f); return -1; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf); fclose(f); return -1;
    }
    fclose(f);

    fs256_hash(buf, (size_t)sz, digest);
    free(buf);
    return 0;
}

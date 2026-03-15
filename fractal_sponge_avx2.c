/*
 * FractalSponge-256 — AVX2 accelerated permutation
 *
 * Compile with: -mavx2 -ffp-contract=off  (no fast-math, no FMA)
 *
 * Strategy: the χ_F step calls fractal_sbox() 8 times with independent inputs.
 * We process 4 S-boxes per AVX2 pass (2 passes per round), computing all
 * three fractal orbits (Julia, Newton, Burning-Ship) 4-wide in double-precision.
 *
 * IEEE 754 strict mode is preserved:
 *   - No FMA intrinsics (_mm256_fmadd_pd etc. are NOT used)
 *   - All mul+add pairs are explicit separate intrinsics
 *   - sqrt uses _mm256_sqrt_pd (same IEEE rounding as scalar sqrt)
 *   - Compiled with -ffp-contract=off to prevent compiler-inserted FMA
 */

#include "fractal_sponge.h"
#include <immintrin.h>   /* AVX2 */
#include <string.h>
#include <stdint.h>

/* ── helpers ─────────────────────────────────────────────────────────────── */

/* 64-bit rotate left, constant shift — AVX2 has no dedicated vprotq */
#define ROT64V(v, n) \
    _mm256_or_si256(_mm256_slli_epi64((v), (n)), _mm256_srli_epi64((v), 64-(n)))

/* 64-bit multiply — keep low 64 bits.
 * AVX2 only has 32×32→64 (mul_epu32); emulate 64×64 lo64:
 *   (a_hi<<32 + a_lo) * (b_hi<<32 + b_lo)  mod 2^64
 *   = a_lo*b_lo + (a_hi*b_lo + a_lo*b_hi) << 32
 */
static inline __m256i mul64_lo(__m256i a, __m256i b) {
    __m256i a_hi   = _mm256_srli_epi64(a, 32);
    __m256i b_hi   = _mm256_srli_epi64(b, 32);
    __m256i lo_lo  = _mm256_mul_epu32(a, b);           /* a_lo * b_lo */
    __m256i hi_lo  = _mm256_mul_epu32(a_hi, b);        /* a_hi * b_lo */
    __m256i lo_hi  = _mm256_mul_epu32(a, b_hi);        /* a_lo * b_hi */
    __m256i cross  = _mm256_add_epi64(hi_lo, lo_hi);
    cross          = _mm256_slli_epi64(cross, 32);
    return           _mm256_add_epi64(lo_lo, cross);
}

/* Convert 4 × uint32 (stored as low 32 bits of each 64-bit lane) to __m256d.
 * Uses the "2^52 magic" trick — no signed conversion artefacts. */
static inline __m256d u32_to_pd(__m256i v32_in_64) {
    /* OR uint32 bits into mantissa of 2^52 → value = 2^52 + uint32 */
    __m256i magic_bits = _mm256_set1_epi64x(0x4330000000000000LL); /* bit-pattern of 2^52 */
    __m256i blended    = _mm256_or_si256(v32_in_64, magic_bits);
    __m256d magic_d    = _mm256_set1_pd(4503599627370496.0);       /* 2^52 as double */
    return _mm256_sub_pd(_mm256_castsi256_pd(blended), magic_d);
}

/* ARX whitening + Murmur3 finalizer — 4-wide */
static inline __m256i arx_whiten_v(__m256i v) {
    __m256i w0 = _mm256_set1_epi64x((long long)ARX_W0);
    __m256i w1 = _mm256_set1_epi64x((long long)ARX_W1);
    __m256i c0 = _mm256_set1_epi64x((long long)0xff51afd7ed558ccdULL);
    __m256i c1 = _mm256_set1_epi64x((long long)0xc4ceb9fe1a85ec53ULL);

    v = _mm256_add_epi64(v, w0);
    v = _mm256_xor_si256(ROT64V(v, 13), w1);
    v = _mm256_add_epi64(v, ROT64V(v, 32));
    v = _mm256_xor_si256(v, _mm256_srli_epi64(v, 33));
    v = mul64_lo(v, c0);
    v = _mm256_xor_si256(v, _mm256_srli_epi64(v, 33));
    v = mul64_lo(v, c1);
    v = _mm256_xor_si256(v, _mm256_srli_epi64(v, 33));
    return v;
}

/* mix64 finalizer — 4-wide */
static inline __m256i mix64_v(__m256i v) {
    __m256i m0 = _mm256_set1_epi64x((long long)MIX_C0);
    __m256i m1 = _mm256_set1_epi64x((long long)MIX_C1);
    v = mul64_lo(_mm256_xor_si256(v, _mm256_srli_epi64(v, 30)), m0);
    v = mul64_lo(_mm256_xor_si256(v, _mm256_srli_epi64(v, 27)), m1);
    return _mm256_xor_si256(v, _mm256_srli_epi64(v, 31));
}

/*
 * fractal_sbox_x4 — compute 4 independent S-boxes simultaneously.
 *
 * words[4]  : the 4 input words (context-mixed by caller)
 * cells[4]  : cell indices (0-3) — selects Julia/Ship parameters per lane
 * rcc4[4]   : per-lane round-constant contributions
 * out[4]    : output words
 */
static void fractal_sbox_x4(const uint64_t words[4],
                             const int      cells[4],
                             const uint64_t rcc4[4],
                             uint64_t       out[4])
{
    /* ── 1. split each word into lo32, hi32, mix ─────────────────────────── */
    __m256i wv    = _mm256_loadu_si256((const __m256i *)words);
    __m256i mask32 = _mm256_set1_epi64x(0xFFFFFFFFLL);

    __m256i lo32v  = _mm256_and_si256(wv, mask32);
    __m256i hi32v  = _mm256_srli_epi64(wv, 32);

    /* rot32(hi32, 13): rotate within 32-bit portion stored in low half of lane */
    __m256i hi_shl = _mm256_and_si256(_mm256_slli_epi64(hi32v, 13), mask32);
    __m256i hi_shr = _mm256_srli_epi64(hi32v, 19);
    __m256i hi_rot = _mm256_or_si256(hi_shl, hi_shr);
    __m256i mixv   = _mm256_xor_si256(lo32v, hi_rot);

    /* ── 2. uint32 → double, scale to [-2, 2) ───────────────────────────── */
    const double SCALE32 = 4.0 / (double)(1ULL << 32);
    __m256d sc    = _mm256_set1_pd(SCALE32);
    __m256d two   = _mm256_set1_pd(2.0);

    __m256d f0 = _mm256_sub_pd(_mm256_mul_pd(u32_to_pd(lo32v), sc), two);
    __m256d f1 = _mm256_sub_pd(_mm256_mul_pd(u32_to_pd(hi32v), sc), two);
    __m256d f2 = _mm256_sub_pd(_mm256_mul_pd(u32_to_pd(mixv),  sc), two);

    /* ── 3. per-lane rc_f scalar: (rcc & 0xFFFFF) / 2^20 - 0.5 ─────────── */
    double rc_f[4];
    for (int k = 0; k < 4; k++)
        rc_f[k] = (double)(rcc4[k] & 0xFFFFF) / (double)(1 << 20) - 0.5;

    /* ── 4. Julia parameters per lane ───────────────────────────────────── */
    __m256d jcre = _mm256_set_pd(
        JC_RE[cells[3]] + rc_f[3]*0.1,
        JC_RE[cells[2]] + rc_f[2]*0.1,
        JC_RE[cells[1]] + rc_f[1]*0.1,
        JC_RE[cells[0]] + rc_f[0]*0.1);
    __m256d jcim = _mm256_set_pd(
        JC_IM[cells[3]] + rc_f[3]*0.07,
        JC_IM[cells[2]] + rc_f[2]*0.07,
        JC_IM[cells[1]] + rc_f[1]*0.07,
        JC_IM[cells[0]] + rc_f[0]*0.07);

    /* ── 5. Julia orbit — 8 iterations, 4 lanes ─────────────────────────── */
    __m256d jre = f0, jim = f1;
    __m256d four = _mm256_set1_pd(4.0);
    __m256d half = _mm256_set1_pd(0.5);
    __m256d one  = _mm256_set1_pd(1.0);

    for (int i = 0; i < 8; i++) {
        __m256d re2  = _mm256_mul_pd(jre, jre);
        __m256d im2  = _mm256_mul_pd(jim, jim);
        __m256d new_re = _mm256_add_pd(_mm256_sub_pd(re2, im2), jcre);
        /* (2.0*jre)*jim matches C left-to-right: 2.0*re*im = (2.0*re)*im */
        __m256d new_im = _mm256_add_pd(
            _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(2.0), jre), jim),
            jcim);
        jre = new_re; jim = new_im;

        __m256d m    = _mm256_add_pd(_mm256_mul_pd(jre,jre), _mm256_mul_pd(jim,jim));
        __m256d bail = _mm256_cmp_pd(m, four, _CMP_GT_OQ);
        /* s = sqrt(m)*0.5; where not bailing s_safe=1 so divide is no-op */
        __m256d s    = _mm256_mul_pd(_mm256_sqrt_pd(m), half);
        __m256d s_safe = _mm256_blendv_pd(one, s, bail);
        jre = _mm256_div_pd(jre, s_safe);
        jim = _mm256_div_pd(jim, s_safe);
    }

    /* ── 6. Newton orbit — 7 iterations, 4 lanes ────────────────────────── */
    __m256d nre = f1, nim = f2;
    __m256d eps  = _mm256_set1_pd(1e-14);
    __m256d sixteen = _mm256_set1_pd(16.0);
    __m256d qtr  = _mm256_set1_pd(0.25);

    for (int i = 0; i < 7; i++) {
        __m256d re2  = _mm256_mul_pd(nre, nre);
        __m256d im2  = _mm256_mul_pd(nim, nim);
        __m256d r2   = _mm256_sub_pd(re2, im2);
        /* (2.0*nre)*nim = C left-to-right: 2.0*re*im */
        __m256d ri   = _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(2.0), nre), nim);
        __m256d r3   = _mm256_sub_pd(_mm256_mul_pd(nre, r2), _mm256_mul_pd(nim, ri));
        __m256d i3   = _mm256_add_pd(_mm256_mul_pd(nre, ri), _mm256_mul_pd(nim, r2));
        __m256d dre  = _mm256_mul_pd(_mm256_set1_pd(3.0), r2);
        /* (6.0*nre)*nim = C left-to-right: 6.0*re*im */
        __m256d dim  = _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(6.0), nre), nim);
        __m256d den  = _mm256_add_pd(_mm256_mul_pd(dre, dre), _mm256_mul_pd(dim, dim));

        /* freeze lanes where den < 1e-14 */
        __m256d frz  = _mm256_cmp_pd(den, eps, _CMP_LT_OQ);
        __m256d den_s = _mm256_blendv_pd(den, one, frz);

        __m256d nr   = _mm256_sub_pd(r3, one);   /* r3 - 1 */
        __m256d ni   = i3;
        /* update = (nr*dre + ni*dim)/den,  (ni*dre - nr*dim)/den */
        __m256d upre = _mm256_div_pd(
            _mm256_add_pd(_mm256_mul_pd(nr, dre), _mm256_mul_pd(ni, dim)), den_s);
        __m256d upim = _mm256_div_pd(
            _mm256_sub_pd(_mm256_mul_pd(ni, dre), _mm256_mul_pd(nr, dim)), den_s);

        __m256d cnre = _mm256_sub_pd(nre, upre);
        __m256d cnim = _mm256_sub_pd(nim, upim);
        /* only apply update where not frozen */
        nre = _mm256_blendv_pd(cnre, nre, frz);
        nim = _mm256_blendv_pd(cnim, nim, frz);

        /* bail-out normalization */
        __m256d m    = _mm256_add_pd(_mm256_mul_pd(nre,nre), _mm256_mul_pd(nim,nim));
        __m256d bail = _mm256_cmp_pd(m, sixteen, _CMP_GT_OQ);
        __m256d s    = _mm256_mul_pd(_mm256_sqrt_pd(m), qtr);
        __m256d s_safe = _mm256_blendv_pd(one, s, bail);
        nre = _mm256_div_pd(nre, s_safe);
        nim = _mm256_div_pd(nim, s_safe);
    }

    /* ── 7. Burning Ship orbit — 8 iterations, 4 lanes ─────────────────── */
    __m256d scre = _mm256_set_pd(
        SC_RE[cells[3]] + rc_f[3]*0.05,
        SC_RE[cells[2]] + rc_f[2]*0.05,
        SC_RE[cells[1]] + rc_f[1]*0.05,
        SC_RE[cells[0]] + rc_f[0]*0.05);
    __m256d scim = _mm256_set_pd(
        SC_IM[cells[3]] + rc_f[3]*0.03,
        SC_IM[cells[2]] + rc_f[2]*0.03,
        SC_IM[cells[1]] + rc_f[1]*0.03,
        SC_IM[cells[0]] + rc_f[0]*0.03);

    __m256d sre = f0, sim = f2;
    /* fabs mask: AND with 0x7FFFFFFFFFFFFFFF clears sign bit */
    __m256d abs_mask = _mm256_castsi256_pd(
        _mm256_set1_epi64x((long long)0x7FFFFFFFFFFFFFFFLL));

    for (int i = 0; i < 8; i++) {
        __m256d are  = _mm256_and_pd(sre, abs_mask);
        __m256d aim  = _mm256_and_pd(sim, abs_mask);
        __m256d are2 = _mm256_mul_pd(are, are);
        __m256d aim2 = _mm256_mul_pd(aim, aim);
        __m256d new_re = _mm256_add_pd(_mm256_sub_pd(are2, aim2), scre);
        /* (2.0*are)*aim matches C left-to-right: 2.0*re*im */
        __m256d new_im = _mm256_add_pd(
            _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(2.0), are), aim),
            scim);
        sre = new_re; sim = new_im;

        __m256d m    = _mm256_add_pd(_mm256_mul_pd(sre,sre), _mm256_mul_pd(sim,sim));
        __m256d bail = _mm256_cmp_pd(m, four, _CMP_GT_OQ);
        __m256d s    = _mm256_mul_pd(_mm256_sqrt_pd(m), half);
        __m256d s_safe = _mm256_blendv_pd(one, s, bail);
        sre = _mm256_div_pd(sre, s_safe);
        sim = _mm256_div_pd(sim, s_safe);
    }

    /* ── 8. Harvest: reinterpret orbit endpoints as uint64, ARX fold ─────── */
    __m256i h0 = _mm256_castpd_si256(jre);
    __m256i h1 = _mm256_castpd_si256(jim);
    __m256i h2 = _mm256_castpd_si256(nre);
    __m256i h3 = _mm256_castpd_si256(nim);
    __m256i h4 = _mm256_castpd_si256(sre);
    __m256i h5 = _mm256_castpd_si256(sim);

    /* per-lane rcc >> 32 */
    __m256i rccv  = _mm256_loadu_si256((const __m256i *)rcc4);
    __m256i rcc_hi = _mm256_srli_epi64(rccv, 32);

    __m256i res = h0;
    res = _mm256_add_epi64(res, ROT64V(h1, 11));
    res = _mm256_xor_si256(res, ROT64V(h2, 23));
    res = _mm256_add_epi64(res, ROT64V(h3, 37));
    res = _mm256_xor_si256(res, ROT64V(h4, 47));
    res = _mm256_add_epi64(res, ROT64V(h5, 53));
    res = _mm256_xor_si256(res, rcc_hi);

    res = mix64_v(arx_whiten_v(res));

    _mm256_storeu_si256((__m256i *)out, res);
}

/* ── rot64 helper (scalar) ────────────────────────────────────────────────── */
static inline uint64_t rot64a(uint64_t v, int n) {
    return (v << n) | (v >> (64 - n));
}

/* ── round constant (scalar) ─────────────────────────────────────────────── */
static inline uint64_t rca(int r) {
    return (RC_PHI   * (uint64_t)(r + 1))
         ^ (RC_SQRT2 * (uint64_t)(r * 7 + 3))
         ^ (RC_SQRT3 * (uint64_t)(r * 13 + 5));
}

/* ── theta_xor (scalar — cheap, not worth AVX-ing) ──────────────────────── */
static void theta_xor_a(uint64_t s[8], uint64_t rcc) {
    uint64_t p  = s[0]^s[1]^s[2]^s[3];
    uint64_t q  = s[4]^s[5]^s[6]^s[7];
    uint64_t d  = p ^ q ^ rot64a(p,1) ^ rot64a(q,7);
    for (int i = 0; i < 8; i++) s[i] ^= d;

    uint64_t p2 = s[0]^s[2]^s[4]^s[6];
    uint64_t q2 = s[1]^s[3]^s[5]^s[7];
    uint64_t d2 = rot64a(p2,13) ^ rot64a(q2,41) ^ rcc;
    for (int i = 0; i < 8; i++) s[i] ^= d2;
}

/*
 * fractal_permutation_avx2 — drop-in replacement for the scalar version.
 *
 * Identical algorithm; only the χ_F S-box evaluation is vectorised.
 * Produces bit-identical output to the scalar path (same IEEE 754 ops,
 * same order, same rounding — no FMA, no fast-math).
 */
void fractal_permutation_avx2(uint64_t s[8]) {
    for (int r = 0; r < FS_ROUNDS; r++) {
        uint64_t rcc = rca(r);

        /* θ_XOR */
        theta_xor_a(s, rcc);

        /* ρ — rotate each word */
        for (int i = 0; i < 8; i++)
            s[i] = rot64a(s[i], FS_RHO[i]);

        /* π — permute positions */
        uint64_t tmp[8];
        for (int i = 0; i < 8; i++) tmp[FS_PI[i]] = s[i];
        memcpy(s, tmp, 64);

        /* χ_F — build context words, then process as two 4-wide SIMD batches */
        uint64_t t[8];
        memcpy(t, s, 64);

        /* context mixing (scalar — just 8 rotations+xors) */
        uint64_t ctx[8];
        for (int i = 0; i < 8; i++) {
            ctx[i] = t[i]
                   ^ rot64a(t[(i+1)%8], 13)
                   ^ rot64a(t[(i+7)%8], 41)
                   ^ rot64a(t[(i+4)%8], 27);
        }

        /* batch 0: words 0-3 (cells 0,1,2,3) */
        {
            const int    cells0[4] = {0, 1, 2, 3};
            uint64_t rcc0[4] = {
                rot64a(rcc, 0*8), rot64a(rcc, 1*8),
                rot64a(rcc, 2*8), rot64a(rcc, 3*8)
            };
            fractal_sbox_x4(ctx, cells0, rcc0, s);   /* s[0..3] */
        }

        /* batch 1: words 4-7 (cells 0,1,2,3) */
        {
            const int    cells1[4] = {0, 1, 2, 3};
            uint64_t rcc1[4] = {
                rot64a(rcc, 4*8), rot64a(rcc, 5*8),
                rot64a(rcc, 6*8), rot64a(rcc, 7*8)
            };
            fractal_sbox_x4(ctx + 4, cells1, rcc1, s + 4);   /* s[4..7] */
        }

        /* ι */
        s[0] ^= rcc;
        s[1] ^= rot64a(rcc, 32);
    }

    /* Final full-state coupling */
    uint64_t xall = s[0]^s[1]^s[2]^s[3]^s[4]^s[5]^s[6]^s[7];
    for (int i = 0; i < 8; i++)
        s[i] ^= rot64a(xall, i * 7 + 1);
}

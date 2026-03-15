#pragma once
#include <stdint.h>
#include <stddef.h>

/* ── output ── */
#define FS256_DIGEST_BYTES  32
#define FS256_DIGEST_HEX    65   /* 64 hex chars + NUL */

/* ── sponge geometry ── */
#define FS_RATE_WORDS   4
#define FS_STATE_WORDS  8
#define FS_BLOCK_BYTES  32
#define FS_ROUNDS       24

/* ── round constants (three irrational seeds) ── */
#define RC_PHI   UINT64_C(0x9E3779B97F4A7C15)
#define RC_SQRT2 UINT64_C(0x6A09E667F3BCC909)
#define RC_SQRT3 UINT64_C(0xBB67AE8584CAA73B)

/* ── ARX whitening constants ── */
#define ARX_W0  UINT64_C(0x736f6d6570736575)
#define ARX_W1  UINT64_C(0x646f72616e646f6d)
#define MIX_C0  UINT64_C(0xbf58476d1ce4e5b9)
#define MIX_C1  UINT64_C(0x94d049bb133111eb)

/* rotation table */
static const int FS_RHO[8] = {1, 3, 6, 10, 15, 21, 28, 36};

/* pi permutation */
static const int FS_PI[8]  = {3, 6, 1, 4, 7, 2, 5, 0};

/* Julia c-params */
static const double JC_RE[4] = { 0.285,    -0.70176,  0.355,  -0.4   };
static const double JC_IM[4] = { 0.013,     0.3842,   0.355,   0.6   };

/* Burning Ship c-params */
static const double SC_RE[4] = {-1.755,    -1.755,    0.400,  -0.500 };
static const double SC_IM[4] = { 0.028,    -0.028,    0.500,  -0.500 };

/* ── public API ── */
#ifdef __cplusplus
extern "C" {
#endif

/* Hash arbitrary data → 32-byte digest */
void fs256_hash(const uint8_t *data, size_t len, uint8_t digest[FS256_DIGEST_BYTES]);

/* Convenience: hash to hex string (65 bytes including NUL) */
void fs256_hash_hex(const uint8_t *data, size_t len, char hex[FS256_DIGEST_HEX]);

/* Hash a file */
int  fs256_hash_file(const char *path, uint8_t digest[FS256_DIGEST_BYTES]);

#ifdef __cplusplus
} /* extern "C" */
#endif

/* When included from a .cu file, also pull in the C header guard explicitly */
#ifdef __CUDACC__
/* nothing extra needed — extern "C" above handles it */
#endif

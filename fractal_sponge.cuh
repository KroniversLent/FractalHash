#pragma once
#include <stdint.h>
#include <stddef.h>
#include <cuda_runtime.h>

/* shared constants — must match fractal_sponge.h exactly */
#define FS256_DIGEST_BYTES  32
#define FS_ROUNDS           24
#define FS_RATE_WORDS        4
#define FS_BLOCK_BYTES      32

#define RC_PHI   UINT64_C(0x9E3779B97F4A7C15)
#define RC_SQRT2 UINT64_C(0x6A09E667F3BCC909)
#define RC_SQRT3 UINT64_C(0xBB67AE8584CAA73B)
#define ARX_W0   UINT64_C(0x736f6d6570736575)
#define ARX_W1   UINT64_C(0x646f72616e646f6d)

/* device functions exposed to other .cu files */
__device__ void d_hash8(const uint8_t msg[8], uint8_t digest[32]);
__device__ void d_hash_var(const uint8_t *msg, int len, uint8_t digest[32]);

/* CUDA error check macro */
#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

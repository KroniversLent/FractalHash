/*
 * birthday.cu — GPU-parallel birthday collision search
 *
 * Strategy:
 *   Phase 1 (GPU): hash N random 8-byte messages, store lower 32/48 bits
 *                  of each digest in a device array.
 *   Phase 2 (CPU): sort the prefix array and scan for adjacent duplicates.
 *
 * N = 2^20 (1M) fits comfortably in GPU memory.
 * N = 2^24 (16M) is feasible on a 4GB+ GPU (128MB for 32-bit prefixes).
 * N = 2^28 (256M) requires ~1GB for 32-bit prefixes — run with --large.
 *
 * Birthday bound: E[collisions] = N^2 / (2 * 2^prefix_bits)
 *   32-bit prefix, N=2^20: E = 2^40 / 2^33 = 2^7 = 128 collisions  ✓ expect many
 *   48-bit prefix, N=2^20: E = 2^40 / 2^49 = 2^-9 ≈ 0              ✓ expect zero
 */

#include "fractal_sponge.cuh"
#include "fractal_sponge.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <algorithm>

/* ── kernel: hash N messages, store 64-bit prefix of each digest ─────────── */

__global__
void birthday_hash_kernel(
    uint64_t  seed,          /* RNG seed — varied per run */
    uint64_t *prefixes,      /* output: one 64-bit value per thread */
    uint32_t *msg_indices,   /* output: thread index (for collision lookup) */
    uint32_t  N              /* total messages to hash */
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    /* Generate deterministic 8-byte message from (seed XOR idx) */
    /* Using a simple bijection so we can reconstruct the message later */
    uint64_t msg_int = seed ^ ((uint64_t)idx * UINT64_C(0x9E3779B97F4A7C15));
    /* Additional mixing so sequential indices give varied messages */
    msg_int ^= msg_int >> 33;
    msg_int *= UINT64_C(0xff51afd7ed558ccd);
    msg_int ^= msg_int >> 33;

    uint8_t msg[8];
    for (int b = 7; b >= 0; b--) {
        msg[b] = msg_int & 0xff;
        msg_int >>= 8;
    }

    uint8_t digest[32];
    d_hash8(msg, digest);

    /* Extract the top 8 bytes of digest as a 64-bit prefix */
    uint64_t prefix = 0;
    for (int b = 0; b < 8; b++)
        prefix = (prefix << 8) | digest[b];

    prefixes[idx]     = prefix;
    msg_indices[idx]  = idx;
}

/* ── host: sort + scan for collisions ────────────────────────────────────── */

typedef struct {
    uint64_t prefix;
    uint32_t idx;
} PrefixEntry;

static int cmp_prefix(const void *a, const void *b) {
    const PrefixEntry *pa = (const PrefixEntry*)a;
    const PrefixEntry *pb = (const PrefixEntry*)b;
    if (pa->prefix < pb->prefix) return -1;
    if (pa->prefix > pb->prefix) return  1;
    return 0;
}

/* Reconstruct message from index + seed */
static void idx_to_msg(uint64_t seed, uint32_t idx, uint8_t msg[8]) {
    uint64_t v = seed ^ ((uint64_t)idx * UINT64_C(0x9E3779B97F4A7C15));
    v ^= v >> 33;
    v *= UINT64_C(0xff51afd7ed558ccd);
    v ^= v >> 33;
    for (int b = 7; b >= 0; b--) { msg[b] = v & 0xff; v >>= 8; }
}

/* ── main birthday test ───────────────────────────────────────────────────── */

void run_birthday_gpu(uint32_t N, int prefix_bits, uint64_t seed) {
    printf("\n=== Birthday test (GPU) ===\n");
    printf("  N = %u (2^%.1f)\n", N, log2((double)N));
    printf("  prefix bits = %d\n", prefix_bits);
    double expected = (double)N*(double)N / (2.0 * pow(2.0, prefix_bits));
    printf("  expected collisions = %.1f\n", expected);

    /* ── Phase 1: GPU hashing ── */
    uint64_t *d_prefixes;
    uint32_t *d_indices;
    CUDA_CHECK(cudaMalloc(&d_prefixes, (size_t)N * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_indices,  (size_t)N * sizeof(uint32_t)));

    int block = 256;
    int grid  = (N + block - 1) / block;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    birthday_hash_kernel<<<grid, block>>>(seed, d_prefixes, d_indices, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double gpu_time = (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)*1e-9;
    printf("  GPU hashing: %.3fs  (%.2fM hashes/sec)\n",
           gpu_time, N/gpu_time/1e6);

    /* ── Phase 2: copy to host and sort ── */
    uint64_t *h_prefixes = (uint64_t*)malloc(N * sizeof(uint64_t));
    uint32_t *h_indices  = (uint32_t*)malloc(N * sizeof(uint32_t));
    CUDA_CHECK(cudaMemcpy(h_prefixes, d_prefixes, N*sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_indices,  d_indices,  N*sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaFree(d_prefixes);
    cudaFree(d_indices);

    /* Apply prefix mask */
    uint64_t mask = (prefix_bits >= 64) ? ~UINT64_C(0)
                                        : ((UINT64_C(1) << prefix_bits) - 1);
    mask <<= (64 - prefix_bits);   /* top prefix_bits bits */

    PrefixEntry *entries = (PrefixEntry*)malloc(N * sizeof(PrefixEntry));
    for (uint32_t i = 0; i < N; i++) {
        entries[i].prefix = h_prefixes[i] & mask;
        entries[i].idx    = h_indices[i];
    }
    free(h_prefixes);
    free(h_indices);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    qsort(entries, N, sizeof(PrefixEntry), cmp_prefix);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double sort_time = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
    printf("  CPU sort:    %.3fs\n", sort_time);

    /* ── Phase 3: scan for collisions ── */
    int n_colls = 0;
    int shown   = 0;
    for (uint32_t i = 1; i < N; i++) {
        if (entries[i].prefix == entries[i-1].prefix) {
            n_colls++;
            if (shown < 3) {
                uint8_t m1[8], m2[8], d1[32], d2[32];
                idx_to_msg(seed, entries[i-1].idx, m1);
                idx_to_msg(seed, entries[i  ].idx, m2);
                /* Recompute on CPU for display — declared in fractal_sponge.h */
                fs256_hash(m1, 8, d1);
                fs256_hash(m2, 8, d2);
                /* Hamming distance */
                int hd = 0;
                for (int b = 0; b < 32; b++)
                    hd += __builtin_popcount(d1[b]^d2[b]);
                printf("  collision %d:\n", n_colls);
                printf("    msg1: "); for(int b=0;b<8;b++) printf("%02x",m1[b]); printf("\n");
                printf("    msg2: "); for(int b=0;b<8;b++) printf("%02x",m2[b]); printf("\n");
                printf("    full-hash hamming: %d bits\n", hd);
                shown++;
            }
        }
    }
    free(entries);

    double ratio = (expected > 0) ? (double)n_colls / expected : 0;
    const char *status = (expected < 1.0 && n_colls == 0) ? "PASS"
                       : (ratio >= 0.5 && ratio <= 2.0)    ? "PASS"
                       : "FAIL";
    printf("  actual collisions:   %d\n", n_colls);
    printf("  ratio actual/exp:    %.2f  [%s]\n", ratio, status);
}

/* ── entry point (called from gpu_main.cu) ───────────────────────────────── */

void birthday_suite_gpu(void) {
    uint64_t seed = (uint64_t)time(NULL);

    /* Three tiers matching the Python test suite */
    run_birthday_gpu(1<<14, 24, seed);   /* sanity:  expect ~8   */
    run_birthday_gpu(1<<20, 32, seed);   /* main:    expect ~122 */
    run_birthday_gpu(1<<20, 48, seed);   /* strength: expect ~0  */
    run_birthday_gpu(1<<24, 32, seed);   /* extended: expect ~2048, takes ~10s */
}

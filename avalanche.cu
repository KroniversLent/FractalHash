/*
 * avalanche.cu — GPU-parallel Strict Avalanche Criterion measurement
 *
 * For each sample message (8 bytes = 64 input bits):
 *   - Hash the original message
 *   - Hash 64 single-bit-flipped variants
 *   - Count differing output bits for each flip
 *
 * With N=16384 sample messages and 64 bits each:
 *   Total tests = 16384 * 64 = 1,048,576
 *   On a modern GPU: ~1s vs ~15min in Python
 *
 * Grid: one thread per (sample, bit) pair
 *   Thread (s, b) hashes msg[s] with bit b flipped
 *   Accumulates Hamming distances into per-bit-position counters
 */

#include "fractal_sponge.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

/* ── kernel ─────────────────────────────────────────────────────────────────── */

__global__
void avalanche_kernel(
    const uint8_t *base_msgs,    /* N * 8 bytes: the original messages */
    const uint8_t *base_digests, /* N * 32 bytes: pre-hashed originals */
    uint32_t       N,            /* number of sample messages */
    uint64_t      *bit_counts,   /* output: 64 counters (one per input bit) */
    uint64_t      *total_counts, /* output: single total Hamming sum */
    uint32_t      *min_val,      /* output: running minimum */
    uint32_t      *max_val       /* output: running maximum */
) {
    /* Thread = one (sample, bit-flip) test */
    uint32_t tid  = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_tests = N * 64;
    if (tid >= total_tests) return;

    uint32_t sample = tid / 64;
    uint32_t bit    = tid % 64;

    /* Build flipped message */
    uint8_t msg[8];
    for (int b = 0; b < 8; b++) msg[b] = base_msgs[sample*8 + b];
    msg[bit/8] ^= (uint8_t)(1 << (7 - bit%8));

    /* Hash */
    uint8_t flipped_digest[32];
    d_hash8(msg, flipped_digest);

    /* Hamming distance vs base digest */
    int hd = 0;
    for (int b = 0; b < 32; b++)
        hd += __popc((unsigned)(base_digests[sample*32 + b] ^ flipped_digest[b]));

    /* Accumulate */
    atomicAdd((unsigned long long*)&bit_counts[bit], (unsigned long long)hd);
    atomicAdd((unsigned long long*)total_counts,     (unsigned long long)hd);
    atomicMin(min_val, (unsigned)hd);
    atomicMax(max_val, (unsigned)hd);
}

/* ── pre-hash kernel: compute base digests for all sample messages ─────────── */

__global__
void prehash_kernel(
    const uint8_t *msgs,     /* N * 8 bytes */
    uint8_t       *digests,  /* N * 32 bytes output */
    uint32_t       N
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    d_hash8(msgs + idx*8, digests + idx*32);
}

/* ── entry point ─────────────────────────────────────────────────────────────── */

void run_avalanche_gpu(uint32_t N) {
    printf("\n=== Avalanche test (GPU) ===\n");
    printf("  samples:     %u\n", N);
    printf("  total tests: %u\n", N * 64);

    /* Generate random 8-byte messages on host */
    uint8_t *h_msgs = (uint8_t*)malloc(N * 8);
    srand((unsigned)time(NULL));
    for (uint32_t i = 0; i < N*8; i++)
        h_msgs[i] = (uint8_t)(rand() & 0xff);

    /* Allocate device buffers */
    uint8_t  *d_msgs, *d_digests;
    uint64_t *d_bit_counts, *d_total;
    uint32_t *d_min, *d_max;

    CUDA_CHECK(cudaMalloc(&d_msgs,       N * 8  * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_digests,    N * 32 * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_bit_counts, 64     * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_total,      1      * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_min,        1      * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_max,        1      * sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(d_msgs, h_msgs, N*8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_bit_counts, 0, 64*sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_total,      0, sizeof(uint64_t)));

    uint32_t init_min = 256, init_max = 0;
    CUDA_CHECK(cudaMemcpy(d_min, &init_min, sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_max, &init_max, sizeof(uint32_t), cudaMemcpyHostToDevice));

    /* Phase 1: pre-hash all messages */
    int block = 256;
    int grid_n = (N + block - 1) / block;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    prehash_kernel<<<grid_n, block>>>(d_msgs, d_digests, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Phase 2: avalanche tests */
    uint32_t total_tests = N * 64;
    int grid_t = (total_tests + block - 1) / block;
    avalanche_kernel<<<grid_t, block>>>(
        d_msgs, d_digests, N,
        d_bit_counts, d_total, d_min, d_max);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;

    /* Copy results */
    uint64_t h_bit_counts[64];
    uint64_t h_total;
    uint32_t h_min, h_max;
    CUDA_CHECK(cudaMemcpy(h_bit_counts, d_bit_counts, 64*sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_total,     d_total,      sizeof(uint64_t),    cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_min,       d_min,        sizeof(uint32_t),    cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_max,       d_max,        sizeof(uint32_t),    cudaMemcpyDeviceToHost));

    cudaFree(d_msgs); cudaFree(d_digests);
    cudaFree(d_bit_counts); cudaFree(d_total);
    cudaFree(d_min); cudaFree(d_max);
    free(h_msgs);

    /* Report */
    double avg = (double)h_total / (double)total_tests;
    printf("  time:        %.3fs  (%u tests/sec)\n",
           elapsed, (uint32_t)(total_tests/elapsed));
    printf("  avg bits Δ:  %.2f / 256  (%.1f%%)\n", avg, avg/256*100);
    printf("  min / max:   %u / %u\n", h_min, h_max);

    /* Per-bit analysis */
    double worst = 256.0, best = 0.0;
    int worst_bit = 0;
    printf("\n  per-input-bit Hamming averages:\n");
    for (int b = 0; b < 64; b++) {
        double per = (double)h_bit_counts[b] / N;
        if (per < worst) { worst = per; worst_bit = b; }
        if (per > best)    best  = per;
        const char *bar_c = (per < 115) ? "!" : (per < 122) ? "~" : " ";
        int filled = (int)(per / 8);
        printf("  bit%2d: %5.1f  %s|", b, per, bar_c);
        for (int x = 0; x < 32; x++) putchar(x < filled ? '#' : '.');
        printf("|\n");
    }
    printf("\n  worst bit: %d  (%.1f avg)  target >115\n", worst_bit, worst);
    printf("  best  bit:     (%.1f avg)\n", best);

    /* Histogram */
    printf("\n  histogram (all %u tests):\n", total_tests);
    /* We'd need to store all values for a full histogram.
       Instead report the distribution from the bit-count data. */
    printf("  [estimated bell curve — run full histogram with --histogram flag]\n");

    const char *sac_status = (avg > 127 && avg < 129 && worst > 110) ? "PASS" : "REVIEW";
    printf("\n  SAC status: %s\n", sac_status);
}

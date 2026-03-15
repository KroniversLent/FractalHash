/*
 * gpu_main.cu — entry point for GPU test suite
 *
 * Usage:
 *   fractal_gpu --avalanche [N]     avalanche test with N sample messages (default 16384)
 *   fractal_gpu --birthday          birthday collision suite (3 tiers)
 *   fractal_gpu --birthday-large    adds 2^24 tier
 *   fractal_gpu --bench             throughput benchmark
 *   fractal_gpu -s "string"         hash a string (CPU path)
 *   fractal_gpu [FILE...]           hash files (CPU path)
 */

#include "fractal_sponge.cuh"
#include "fractal_sponge.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* forward declarations */
void run_avalanche_gpu(uint32_t N);
void birthday_suite_gpu(void);
void run_birthday_gpu(uint32_t N, int prefix_bits, uint64_t seed);
void run_differential_analysis(uint32_t N);

/* ── GPU throughput benchmark ─────────────────────────────────────────────── */

__global__
void bench_kernel(uint64_t seed, uint8_t *out, uint32_t N) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    uint8_t msg[8];
    uint64_t v = seed ^ idx;
    for (int b = 7; b >= 0; b--) { msg[b] = v&0xff; v>>=8; }
    d_hash8(msg, out + idx*32);
}

static void gpu_bench(void) {
    const uint32_t N = 1 << 20;  /* 1M hashes */
    uint8_t *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)N * 32));

    /* Warm up */
    bench_kernel<<<(N+255)/256, 256>>>(42ULL, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    bench_kernel<<<(N+255)/256, 256>>>(123ULL, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
    printf("GPU bench: %uM hashes in %.3fs  →  %.2fM hashes/sec\n",
           N>>20, elapsed, N/elapsed/1e6);
    cudaFree(d_out);
}

/* ── device info ──────────────────────────────────────────────────────────── */

static void print_device_info(void) {
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);
    printf("GPU: %s  (sm_%d%d)  %.0f GB  %d SMs\n",
           p.name, p.major, p.minor,
           p.totalGlobalMem / 1e9,
           p.multiProcessorCount);
}

/* ── main ─────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    print_device_info();

    if (argc < 2) {
        printf("Usage:\n");
        printf("  fractal_gpu --differential [N]  differential cryptanalysis (default N=1M)\n");
        printf("  fractal_gpu --avalanche [N]     avalanche test (default N=16384)\n");
        printf("  fractal_gpu --birthday          birthday collision suite\n");
        printf("  fractal_gpu --bench             throughput benchmark\n");
        printf("  fractal_gpu -s \"string\"         hash string (CPU)\n");
        printf("  fractal_gpu FILE ...             hash files (CPU)\n");
        return 0;
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--differential") == 0) {
            uint32_t N = 1 << 20;
            if (i+1 < argc && argv[i+1][0] != '-')
                N = (uint32_t)atoi(argv[++i]);
            run_differential_analysis(N);

        } else if (strcmp(argv[i], "--avalanche") == 0) {
            uint32_t N = 16384;
            if (i+1 < argc && argv[i+1][0] != '-')
                N = (uint32_t)atoi(argv[++i]);
            run_avalanche_gpu(N);

        } else if (strcmp(argv[i], "--birthday") == 0) {
            birthday_suite_gpu();

        } else if (strcmp(argv[i], "--birthday-large") == 0) {
            uint64_t seed = (uint64_t)time(NULL);
            run_birthday_gpu(1<<24, 32, seed);

        } else if (strcmp(argv[i], "--bench") == 0) {
            gpu_bench();

        } else if (strcmp(argv[i], "-s") == 0 && i+1 < argc) {
            const char *s = argv[++i];
            char hex[65];
            fs256_hash_hex((const uint8_t*)s, strlen(s), hex);
            printf("%s  \"%s\"\n", hex, s);

        } else {
            /* File hash via CPU */
            uint8_t digest[32];
            char hex[65];
            if (fs256_hash_file(argv[i], digest) != 0) {
                fprintf(stderr, "cannot open: %s\n", argv[i]);
                continue;
            }
            static const char *hc = "0123456789abcdef";
            for (int j = 0; j < 32; j++) {
                hex[2*j]   = hc[digest[j]>>4];
                hex[2*j+1] = hc[digest[j]&0xf];
            }
            hex[64] = '\0';
            printf("%s  %s\n", hex, argv[i]);
        }
    }
    return 0;
}

/*
 * differential.cu — GPU differential cryptanalysis of FractalSponge-256
 *
 * Three tests:
 *
 * 1. INPUT DIFFERENTIAL DISTRIBUTION
 *    For each of 64 single-bit input differentials ΔM = e_i:
 *    - Hash N random messages M and M⊕ΔM
 *    - Build a histogram of output differentials ΔH = H(M) ⊕ H(M⊕ΔM)
 *    - Test: does any output differential appear more than expected?
 *    - A random oracle gives each 256-bit ΔH with probability 2^-256
 *    - We compress ΔH to 32-bit fingerprint for tractable counting
 *    - Expected: flat histogram. Any spike > 3σ above mean is a bias.
 *
 * 2. OUTPUT DIFFERENTIAL BIAS MATRIX
 *    For each input bit i and output bit j:
 *    - P(output_bit_j_flips | input_bit_i_flips)
 *    - Should be 0.5 for all (i,j). Bias = |P - 0.5|
 *    - Build full 64×256 matrix (16384 entries)
 *    - Report worst-case and distribution of biases
 *
 * 3. HIGH-ORDER DIFFERENTIAL — 2-BIT INPUT DIFFERENTIALS
 *    For pairs of input bits (i,j): ΔM = e_i ⊕ e_j
 *    Tests whether 2-bit input changes show non-random output patterns
 *    There are C(64,2) = 2016 such pairs
 *
 * What we're looking for (in order of severity):
 *   - Any output differential appearing > 2^16 times in 2^24 trials: CRITICAL
 *   - Any (input_bit, output_bit) bias > 0.01: SIGNIFICANT
 *   - Any 2-bit differential with avg output Hamming < 120 or > 136: NOTABLE
 *   - Everything within 3σ of expected: PASS
 */

#include "fractal_sponge.cuh"
#include "fractal_sponge.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ── Test 1: Input differential → output differential histogram ──────────── */

/*
 * For input differential delta_in (a single flipped bit position 0..63),
 * hash N random messages and their differentially-modified counterparts.
 * Accumulate output differential fingerprints into a 2^16-bucket histogram.
 * Fingerprint = top 16 bits of (ΔH[0..7] XOR ΔH[8..15] XOR ΔH[16..23] XOR ΔH[24..31])
 * This compresses 256-bit ΔH to 16 bits while preserving structure.
 */
__global__
void diff_histogram_kernel(
    uint64_t   seed,
    uint8_t    delta_byte,     /* which byte of msg to flip */
    uint8_t    delta_bit_mask, /* which bit within that byte */
    uint32_t  *histogram,      /* 2^16 buckets, atomically updated */
    uint32_t   N
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    /* Generate message */
    uint64_t v = seed ^ ((uint64_t)idx * UINT64_C(0x9E3779B97F4A7C15));
    v ^= v >> 33; v *= UINT64_C(0xff51afd7ed558ccd); v ^= v >> 33;
    uint8_t msg[8];
    for (int b = 7; b >= 0; b--) { msg[b] = v & 0xff; v >>= 8; }

    /* Compute H(M) */
    uint8_t d1[32];
    d_hash8(msg, d1);

    /* Compute H(M ⊕ ΔM) */
    uint8_t msg2[8];
    for (int b = 0; b < 8; b++) msg2[b] = msg[b];
    msg2[delta_byte] ^= delta_bit_mask;
    uint8_t d2[32];
    d_hash8(msg2, d2);

    /* Output differential ΔH = d1 ⊕ d2, compressed to 16-bit fingerprint */
    uint32_t fp = 0;
    for (int b = 0; b < 32; b++) {
        uint8_t diff_byte = d1[b] ^ d2[b];
        /* XOR-fold into 16 bits with position-dependent rotation */
        fp ^= (uint32_t)diff_byte << ((b * 3) & 8);
    }
    fp &= 0xFFFF;

    atomicAdd(&histogram[fp], 1u);
}

/* ── Test 2: Output differential bias matrix ─────────────────────────────── */

/*
 * For each (input_bit i, output_bit j) pair:
 * Count how many times output bit j flips when input bit i flips.
 * bias_matrix[i * 256 + j] = count (divide by N to get probability).
 * Expected: N/2 for all entries. Deviation = |count - N/2|.
 */
__global__
void diff_bias_kernel(
    uint64_t   seed,
    uint64_t  *bias_matrix,   /* 64 * 256 counters */
    uint32_t   N
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    /* Generate message */
    uint64_t v = seed ^ ((uint64_t)idx * UINT64_C(0x9E3779B97F4A7C15));
    v ^= v >> 33; v *= UINT64_C(0xff51afd7ed558ccd); v ^= v >> 33;
    uint8_t msg[8];
    for (int b = 7; b >= 0; b--) { msg[b] = v & 0xff; v >>= 8; }

    /* Hash base message */
    uint8_t d_base[32];
    d_hash8(msg, d_base);

    /* For each of 64 input bit flips */
    for (int ibit = 0; ibit < 64; ibit++) {
        uint8_t msg2[8];
        for (int b = 0; b < 8; b++) msg2[b] = msg[b];
        msg2[ibit/8] ^= (uint8_t)(1 << (7 - ibit%8));

        uint8_t d_flip[32];
        d_hash8(msg2, d_flip);

        /* Count which output bits flipped */
        for (int obit = 0; obit < 256; obit++) {
            int obyte = obit / 8;
            int obitpos = 7 - (obit % 8);
            int flipped = ((d_base[obyte] ^ d_flip[obyte]) >> obitpos) & 1;
            if (flipped)
                atomicAdd((unsigned long long*)&bias_matrix[ibit * 256 + obit],
                          (unsigned long long)1);
        }
    }
}

/* ── Test 3: 2-bit input differentials ───────────────────────────────────── */

__global__
void diff_2bit_kernel(
    uint64_t   seed,
    uint32_t   ibit_a,        /* first bit to flip */
    uint32_t   ibit_b,        /* second bit to flip */
    uint64_t  *hamming_sum,   /* accumulator for total Hamming distance */
    uint32_t  *hamming_min,
    uint32_t  *hamming_max,
    uint32_t   N
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    uint64_t v = seed ^ ((uint64_t)idx * UINT64_C(0x9E3779B97F4A7C15));
    v ^= v >> 33; v *= UINT64_C(0xff51afd7ed558ccd); v ^= v >> 33;
    uint8_t msg[8];
    for (int b = 7; b >= 0; b--) { msg[b] = v & 0xff; v >>= 8; }

    uint8_t d1[32];
    d_hash8(msg, d1);

    uint8_t msg2[8];
    for (int b = 0; b < 8; b++) msg2[b] = msg[b];
    msg2[ibit_a/8] ^= (uint8_t)(1 << (7 - ibit_a%8));
    msg2[ibit_b/8] ^= (uint8_t)(1 << (7 - ibit_b%8));

    uint8_t d2[32];
    d_hash8(msg2, d2);

    uint32_t hd = 0;
    for (int b = 0; b < 32; b++)
        hd += __popc((unsigned)(d1[b] ^ d2[b]));

    atomicAdd((unsigned long long*)hamming_sum, (unsigned long long)hd);
    atomicMin(hamming_min, hd);
    atomicMax(hamming_max, hd);
}

/* ── Host: run all three tests ───────────────────────────────────────────── */

static double chi2_pvalue(double chi2, int df) {
    /* Regularised incomplete gamma approximation for chi2 survival function */
    double a = df / 2.0, x = chi2 / 2.0;
    if (x <= 0) return 1.0;
    double lna = a * log(x) - x - lgamma(a + 1);
    double s = 1.0, t = 1.0;
    for (int n = 1; n < 300; n++) {
        t *= x / (a + n); s += t;
        if (t < 1e-14) break;
    }
    double p = 1.0 - exp(lna) * s;
    return p < 0 ? 0 : p > 1 ? 1 : p;
}

void run_differential_analysis(uint32_t N) {
    printf("\n========================================\n");
    printf("Differential cryptanalysis (GPU)\n");
    printf("========================================\n");
    printf("N = %u samples per test\n\n", N);

    uint64_t seed = (uint64_t)time(NULL);
    int block = 256;

    /* ── Test 1: Differential histogram for all 64 input bits ── */
    printf("--- Test 1: Output differential histogram ---\n");
    printf("    For each of 64 input-bit differentials, check if any\n");
    printf("    output differential fingerprint appears anomalously often.\n");
    printf("    Expected: flat histogram over 2^16 buckets.\n");
    printf("    Threshold: count > mean + 6σ is suspicious.\n\n");

    uint32_t *d_hist;
    CUDA_CHECK(cudaMalloc(&d_hist, (1<<16) * sizeof(uint32_t)));

    double worst_chi2 = 0;
    int    worst_ibit = -1;
    uint32_t worst_max_count = 0;

    for (int ibit = 0; ibit < 64; ibit++) {
        CUDA_CHECK(cudaMemset(d_hist, 0, (1<<16) * sizeof(uint32_t)));
        int grid = (N + block - 1) / block;
        diff_histogram_kernel<<<grid, block>>>(
            seed + ibit,
            (uint8_t)(ibit / 8),
            (uint8_t)(1 << (7 - ibit % 8)),
            d_hist, N
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        uint32_t *h_hist = (uint32_t*)malloc((1<<16) * sizeof(uint32_t));
        CUDA_CHECK(cudaMemcpy(h_hist, d_hist, (1<<16)*sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));

        /* Compute chi-squared over histogram */
        double expected = (double)N / (1<<16);
        double chi2 = 0;
        uint32_t max_count = 0;
        uint32_t nonzero = 0;
        for (int b = 0; b < (1<<16); b++) {
            if (h_hist[b] > 0) nonzero++;
            if (h_hist[b] > max_count) max_count = h_hist[b];
            double diff = (double)h_hist[b] - expected;
            chi2 += diff * diff / expected;
        }
        free(h_hist);

        double p = chi2_pvalue(chi2, (1<<16) - 1);
        /* sigma above mean for max count */
        double mean = expected;
        double sigma = sqrt(expected);
        double zscore = (max_count - mean) / sigma;

        if (chi2 > worst_chi2) {
            worst_chi2 = chi2;
            worst_ibit = ibit;
            worst_max_count = max_count;
        }

        char flag = (zscore > 6.0 || p < 0.001) ? '!' : ' ';
        if (ibit % 8 == 0 || flag == '!')
            printf("  bit%2d: chi2=%8.1f  p=%.4f  max_count=%4u  z=%.2f %c\n",
                   ibit, chi2, p, max_count, zscore, flag);
    }
    cudaFree(d_hist);

    printf("\n  Worst: input bit %d  chi2=%.1f  max_bucket=%u\n",
           worst_ibit, worst_chi2, worst_max_count);
    double expected_chi2 = (double)((1<<16) - 1);  /* df = 2^16-1 */
    printf("  Expected chi2 ~ %.0f  (df=%d)\n", expected_chi2, (1<<16)-1);
    printf("  Result: %s\n\n",
           (worst_chi2 < expected_chi2 * 1.05) ? "PASS — no anomalous differential fingerprints"
                                                : "REVIEW — chi2 elevated");

    /* ── Test 2: Bias matrix ── */
    printf("--- Test 2: Output differential bias matrix ---\n");
    printf("    P(output_bit_j flips | input_bit_i flips) should be 0.5.\n");
    printf("    Checking all 64 × 256 = 16384 (input,output) bit pairs.\n\n");

    uint64_t *d_bias;
    CUDA_CHECK(cudaMalloc(&d_bias, 64 * 256 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_bias, 0, 64 * 256 * sizeof(uint64_t)));

    /* Reduce N for this test since each thread does 64*256 atomics */
    uint32_t N_bias = N / 4;
    int grid_b = (N_bias + block - 1) / block;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    diff_bias_kernel<<<grid_b, block>>>(seed, d_bias, N_bias);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
    printf("  Computed %u × 64 × 256 = %llu bias measurements in %.2fs\n",
           N_bias, (unsigned long long)N_bias * 64 * 256, elapsed);

    uint64_t *h_bias = (uint64_t*)malloc(64 * 256 * sizeof(uint64_t));
    CUDA_CHECK(cudaMemcpy(h_bias, d_bias, 64*256*sizeof(uint64_t),
                          cudaMemcpyDeviceToHost));
    cudaFree(d_bias);

    double max_bias = 0, sum_bias = 0;
    int max_ibit = 0, max_obit = 0;
    uint32_t over_1pct = 0, over_5pct = 0;
    double half = N_bias * 0.5;

    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 256; j++) {
            double cnt = (double)h_bias[i * 256 + j];
            double bias = fabs(cnt - half) / N_bias;
            sum_bias += bias;
            if (bias > max_bias) {
                max_bias = bias; max_ibit = i; max_obit = j;
            }
            if (bias > 0.01) over_1pct++;
            if (bias > 0.05) over_5pct++;
        }
    }
    free(h_bias);

    printf("  max bias:  %.4f  (input bit %d → output bit %d)\n",
           max_bias, max_ibit, max_obit);
    printf("  avg bias:  %.5f  (expected ~%.5f for N=%u)\n",
           sum_bias / (64*256), 1.0/sqrt(M_PI * N_bias / 2.0), N_bias);
    printf("  pairs with bias > 1%%:  %u / 16384\n", over_1pct);
    printf("  pairs with bias > 5%%:  %u / 16384\n", over_5pct);

    /* Statistical threshold: for N_bias samples, σ = 0.5/√N_bias */
    double sigma_bias = 0.5 / sqrt((double)N_bias);
    double threshold_3sig = 3.0 * sigma_bias;
    double threshold_6sig = 6.0 * sigma_bias;
    printf("  3σ threshold: %.4f  |  6σ threshold: %.4f\n",
           threshold_3sig, threshold_6sig);
    printf("  Result: %s\n\n",
           (max_bias < threshold_6sig) ? "PASS — no significant bias found"
           : (max_bias < 0.05)         ? "REVIEW — weak bias detected"
                                       : "FAIL — strong bias detected");

    /* ── Test 3: 2-bit input differentials ── */
    printf("--- Test 3: 2-bit input differentials ---\n");
    printf("    Checking C(64,2)=2016 pairs of input bit flips.\n");
    printf("    Expected avg Hamming: 128.0.  Alarm if |avg-128| > 2.0\n\n");

    uint64_t *d_hsum; uint32_t *d_hmin, *d_hmax;
    CUDA_CHECK(cudaMalloc(&d_hsum, sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_hmin, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_hmax, sizeof(uint32_t)));

    uint32_t N_2bit = N / 4;
    int grid_2 = (N_2bit + block - 1) / block;

    double worst_avg = 128.0;
    int worst_a = -1, worst_b_bit = -1;
    int pairs_checked = 0, pairs_flagged = 0;

    /* Check all 2016 pairs — show progress every 200 */
    for (int a = 0; a < 64; a++) {
        for (int b = a+1; b < 64; b++) {
            uint64_t init_sum = 0; uint32_t init_min = 256, init_max = 0;
            CUDA_CHECK(cudaMemcpy(d_hsum, &init_sum, 8, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_hmin, &init_min, 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_hmax, &init_max, 4, cudaMemcpyHostToDevice));

            diff_2bit_kernel<<<grid_2, block>>>(
                seed + a*64 + b, a, b, d_hsum, d_hmin, d_hmax, N_2bit);
            CUDA_CHECK(cudaDeviceSynchronize());

            uint64_t hsum; uint32_t hmin, hmax;
            CUDA_CHECK(cudaMemcpy(&hsum, d_hsum, 8, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&hmin, d_hmin, 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&hmax, d_hmax, 4, cudaMemcpyDeviceToHost));

            double avg = (double)hsum / N_2bit;
            pairs_checked++;

            if (fabs(avg - 128.0) > 2.0) {
                pairs_flagged++;
                printf("  ! pair (%d,%d): avg=%.2f  min=%u  max=%u\n",
                       a, b, avg, hmin, hmax);
            }
            if (fabs(avg - 128.0) > fabs(worst_avg - 128.0)) {
                worst_avg = avg;
                worst_a = a; worst_b_bit = b;
            }

            if (pairs_checked % 400 == 0)
                printf("  ... %d/2016 pairs checked, %d flagged so far\n",
                       pairs_checked, pairs_flagged);
        }
    }

    cudaFree(d_hsum); cudaFree(d_hmin); cudaFree(d_hmax);

    printf("\n  Checked all %d 2-bit pairs\n", pairs_checked);
    printf("  Worst: bits (%d,%d)  avg=%.2f  (deviation %.2f from 128)\n",
           worst_a, worst_b_bit, worst_avg, fabs(worst_avg-128.0));
    printf("  Flagged (|avg-128|>2.0): %d / 2016\n", pairs_flagged);

    /* Expected number of false positives at |avg-128|>2.0 threshold:
     * σ for avg of N_2bit samples ≈ sqrt(128*128/N_2bit) ≈ 0.14 for N=64K
     * So >2.0 deviation is > 14σ — should never happen randomly */
    double expected_sigma = sqrt(128.0 * 128.0 / N_2bit);
    printf("  σ for avg at N=%u: %.3f  |  2.0 deviation = %.1fσ\n",
           N_2bit, expected_sigma, 2.0 / expected_sigma);
    printf("  Result: %s\n\n",
           (pairs_flagged == 0) ? "PASS — no anomalous 2-bit differentials"
                                : "FAIL — anomalous differential(s) detected");

    printf("========================================\n");
    printf("Differential cryptanalysis summary\n");
    printf("========================================\n");
    printf("  Test 1 (differential histogram):  %s\n",
           worst_chi2 < ((1<<16)-1)*1.05 ? "PASS" : "REVIEW");
    printf("  Test 2 (bias matrix 64×256):      %s\n",
           max_bias < 6.0*(0.5/sqrt((double)(N/4))) ? "PASS" : "REVIEW");
    printf("  Test 3 (2-bit differentials):     %s\n",
           pairs_flagged == 0 ? "PASS" : "FAIL");
    printf("\n  GPU: RTX 3090 Ti — total tests ≈ %llu hash evaluations\n",
           (unsigned long long)(64*N + (uint64_t)(N/4)*64*2 + (uint64_t)2016*(N/4)*2));
}

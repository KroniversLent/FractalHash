/*
 * fractal_hash — CLI tool, sha256sum-compatible output format
 *
 * Usage:
 *   fractal_hash [FILE...]          hash files
 *   fractal_hash -s "string"        hash a string
 *   fractal_hash --test             run built-in test vectors
 *   echo "hello" | fractal_hash     hash stdin
 */

#include "fractal_sponge.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

static void print_digest_file(const char *path) {
    uint8_t digest[FS256_DIGEST_BYTES];
    char    hex[FS256_DIGEST_HEX];
    if (fs256_hash_file(path, digest) != 0) {
        fprintf(stderr, "fractal_hash: %s: cannot open\n", path);
        return;
    }
    fs256_hash_hex(NULL, 0, hex); /* reuse hex buffer — just for compiler */
    static const char *hx = "0123456789abcdef";
    for (int i = 0; i < FS256_DIGEST_BYTES; i++) {
        putchar(hx[digest[i]>>4]);
        putchar(hx[digest[i]&0xf]);
    }
    printf("  %s\n", path);
}

static void print_digest_buf(const uint8_t *data, size_t len, const char *label) {
    char hex[FS256_DIGEST_HEX];
    fs256_hash_hex(data, len, hex);
    printf("%s  %s\n", hex, label);
}

static void stdin_hash(void) {
    uint8_t *buf = NULL;
    size_t   cap = 0, len = 0;
    int      c;
    while ((c = getchar()) != EOF) {
        if (len == cap) {
            cap = cap ? cap*2 : 4096;
            buf = realloc(buf, cap);
        }
        buf[len++] = (uint8_t)c;
    }
    print_digest_buf(buf ? buf : (const uint8_t*)"", len, "-");
    free(buf);
}

static void run_tests(void) {
    struct { const char *input; } cases[] = {
        {""}, {"test"}, {"hello world"}, {"a"},
        {"The quick brown fox jumps over the lazy dog"},
        {NULL}
    };
    printf("%-44s  hash\n", "input");
    printf("%s\n", "------------------------------------------------------------"
                   "----------------------------");
    for (int i = 0; cases[i].input; i++) {
        const char *s = cases[i].input;
        char label[48];
        snprintf(label, sizeof(label), "\"%s\"", s);
        print_digest_buf((const uint8_t*)s, strlen(s), label);
    }

    /* Single-bit sensitivity */
    printf("\nSingle-bit sensitivity on \"test\":\n");
    const char *base_str = "test";
    uint8_t base[FS256_DIGEST_BYTES];
    fs256_hash((const uint8_t*)base_str, 4, base);

    for (int bit = 0; bit < 8; bit++) {
        uint8_t msg[4]; memcpy(msg, base_str, 4);
        msg[bit/8] ^= 1 << (7 - bit%8);
        uint8_t d[FS256_DIGEST_BYTES];
        fs256_hash(msg, 4, d);
        int diff = 0;
        for (int j = 0; j < FS256_DIGEST_BYTES; j++)
            diff += __builtin_popcount(base[j] ^ d[j]);
        printf("  bit %d flipped: %d bits differ\n", bit, diff);
    }
}

static void bench(void) {
    uint8_t data[1024];
    for (int i = 0; i < 1024; i++) data[i] = (uint8_t)i;
    uint8_t digest[FS256_DIGEST_BYTES];

    int iters = 200;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; i++)
        fs256_hash(data, 1024, digest);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)*1e-9;
    printf("CPU bench: %d × 1KB in %.3fs → %.1f KB/s  (%.2f ms/hash)\n",
           iters, elapsed,
           (double)iters / elapsed,
           elapsed / iters * 1000.0);
}

int main(int argc, char **argv) {
    if (argc < 2) { stdin_hash(); return 0; }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--test") == 0)  { run_tests(); }
        else if (strcmp(argv[i], "--bench") == 0) { bench(); }
        else if (strcmp(argv[i], "-s") == 0 && i+1 < argc) {
            const char *s = argv[++i];
            print_digest_buf((const uint8_t*)s, strlen(s), s);
        }
        else { print_digest_file(argv[i]); }
    }
    return 0;
}

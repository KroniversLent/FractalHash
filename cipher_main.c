/*
 * FractalCipher CLI — stream cipher with public/private key exchange
 *
 * Usage:
 *   fractal_cipher keygen  <privkey_file> <pubkey_file>
 *   fractal_cipher shared  <my_privkey_file> <peer_pubkey_file> <shared_out>
 *   fractal_cipher encrypt <shared_file> <input> <output>
 *   fractal_cipher decrypt <shared_file> <input> <output>
 *   fractal_cipher test
 */

#include "fractal_cipher.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* ── file helpers ─────────────────────────────────────────────────────────── */

static int read_file_exact(const char *path, uint8_t *buf, size_t expected) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "error: cannot open '%s'\n", path); return -1; }
    size_t got = fread(buf, 1, expected, f);
    fclose(f);
    if (got != expected) {
        fprintf(stderr, "error: '%s' is %zu bytes, expected %zu\n",
                path, got, expected);
        return -1;
    }
    return 0;
}

static int write_file(const char *path, const uint8_t *buf, size_t len) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "error: cannot create '%s'\n", path); return -1; }
    size_t written = fwrite(buf, 1, len, f);
    fclose(f);
    if (written != len) {
        fprintf(stderr, "error: short write to '%s'\n", path);
        return -1;
    }
    return 0;
}

static uint8_t *read_whole_file(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "error: cannot open '%s'\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    if (sz < 0) { fclose(f); return NULL; }
    uint8_t *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    *len_out = (size_t)sz;
    if (sz > 0 && fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf); fclose(f); return NULL;
    }
    fclose(f);
    return buf;
}

static void print_hex(const char *label, const uint8_t *data, size_t len) {
    printf("%s: ", label);
    for (size_t i = 0; i < len; i++) printf("%02x", data[i]);
    printf("\n");
}

/* ── commands ─────────────────────────────────────────────────────────────── */

static int cmd_keygen(const char *priv_path, const char *pub_path) {
    FcKeypair kp;
    if (fc_keygen(&kp) != 0) {
        fprintf(stderr, "error: keygen failed (cannot read /dev/urandom)\n");
        return 1;
    }
    if (write_file(priv_path, kp.priv, FC_PRIVKEY_BYTES) != 0) return 1;
    if (write_file(pub_path,  kp.pub,  FC_PUBKEY_BYTES)  != 0) return 1;

    printf("Generated keypair:\n");
    print_hex("  private", kp.priv, FC_PRIVKEY_BYTES);
    print_hex("  public ", kp.pub,  FC_PUBKEY_BYTES);
    printf("  private → %s\n  public  → %s\n", priv_path, pub_path);
    return 0;
}

static int cmd_shared(const char *priv_path, const char *peer_pub_path,
                      const char *out_path) {
    uint8_t priv[FC_PRIVKEY_BYTES], peer_pub[FC_PUBKEY_BYTES];
    if (read_file_exact(priv_path,     priv,     FC_PRIVKEY_BYTES) != 0) return 1;
    if (read_file_exact(peer_pub_path, peer_pub, FC_PUBKEY_BYTES)  != 0) return 1;

    uint8_t shared[FC_SHARED_BYTES];
    fc_shared_secret(priv, peer_pub, shared);

    if (write_file(out_path, shared, FC_SHARED_BYTES) != 0) return 1;
    print_hex("Shared secret", shared, FC_SHARED_BYTES);
    printf("  → %s\n", out_path);
    return 0;
}

/* On-disk format for encrypted files:
 *   [16 bytes nonce][16 bytes auth tag][ciphertext...]
 */
#define HEADER_BYTES  (FC_NONCE_BYTES + FC_TAG_BYTES)

static int cmd_encrypt(const char *shared_path, const char *in_path,
                       const char *out_path) {
    uint8_t shared[FC_SHARED_BYTES];
    if (read_file_exact(shared_path, shared, FC_SHARED_BYTES) != 0) return 1;

    size_t pt_len;
    uint8_t *pt = read_whole_file(in_path, &pt_len);
    if (!pt) return 1;

    uint8_t nonce[FC_NONCE_BYTES];
    if (fc_random_nonce(nonce) != 0) {
        fprintf(stderr, "error: cannot generate nonce\n");
        free(pt); return 1;
    }

    uint8_t *ct  = malloc(pt_len);
    uint8_t *out = malloc(HEADER_BYTES + pt_len);
    if (!ct || !out) { free(pt); free(ct); free(out); return 1; }

    uint8_t tag[FC_TAG_BYTES];
    fc_encrypt(shared, nonce, NULL, 0, pt, ct, pt_len, tag);

    /* Pack: nonce || tag || ciphertext */
    memcpy(out,                              nonce, FC_NONCE_BYTES);
    memcpy(out + FC_NONCE_BYTES,             tag,   FC_TAG_BYTES);
    memcpy(out + HEADER_BYTES,               ct,    pt_len);

    int rc = write_file(out_path, out, HEADER_BYTES + pt_len);

    free(pt); free(ct); free(out);
    if (rc != 0) return 1;

    printf("Encrypted: %zu → %zu bytes\n", pt_len, HEADER_BYTES + pt_len);
    print_hex("  nonce", nonce, FC_NONCE_BYTES);
    print_hex("  tag  ", tag,   FC_TAG_BYTES);
    return 0;
}

static int cmd_decrypt(const char *shared_path, const char *in_path,
                       const char *out_path) {
    uint8_t shared[FC_SHARED_BYTES];
    if (read_file_exact(shared_path, shared, FC_SHARED_BYTES) != 0) return 1;

    size_t in_len;
    uint8_t *in_buf = read_whole_file(in_path, &in_len);
    if (!in_buf) return 1;

    if (in_len < HEADER_BYTES) {
        fprintf(stderr, "error: input too short (no header)\n");
        free(in_buf); return 1;
    }

    uint8_t nonce[FC_NONCE_BYTES], tag[FC_TAG_BYTES];
    memcpy(nonce, in_buf,                  FC_NONCE_BYTES);
    memcpy(tag,   in_buf + FC_NONCE_BYTES, FC_TAG_BYTES);

    size_t ct_len = in_len - HEADER_BYTES;
    const uint8_t *ct = in_buf + HEADER_BYTES;
    uint8_t *pt = malloc(ct_len + 1);
    if (!pt) { free(in_buf); return 1; }

    int rc = fc_decrypt(shared, nonce, NULL, 0, ct, pt, ct_len, tag);
    if (rc != 0) {
        fprintf(stderr, "error: authentication failed — wrong key, nonce, or tampered data\n");
        free(in_buf); free(pt);
        return 1;
    }

    if (write_file(out_path, pt, ct_len) != 0) {
        free(in_buf); free(pt); return 1;
    }
    free(in_buf); free(pt);
    printf("Decrypted: %zu → %zu bytes\n", in_len, ct_len);
    return 0;
}

/* ── built-in self-test ───────────────────────────────────────────────────── */

static int cmd_test(void) {
    int pass = 1;

    /* Test 1: keygen + shared secret symmetry */
    printf("Test 1: keypair + shared secret symmetry ... ");
    FcKeypair alice, bob;
    if (fc_keygen(&alice) != 0 || fc_keygen(&bob) != 0) {
        printf("FAIL (keygen error)\n"); return 1;
    }
    uint8_t shared_ab[FC_SHARED_BYTES], shared_ba[FC_SHARED_BYTES];
    fc_shared_secret(alice.priv, bob.pub,   shared_ab);
    fc_shared_secret(bob.priv,   alice.pub, shared_ba);
    if (memcmp(shared_ab, shared_ba, FC_SHARED_BYTES) == 0) {
        printf("PASS\n");
    } else {
        printf("FAIL (shared secrets differ)\n"); pass = 0;
    }

    /* Test 2: encrypt/decrypt round-trip */
    printf("Test 2: encrypt/decrypt round-trip ... ");
    const char *plaintext = "The quick brown fox jumps over the lazy dog";
    size_t pt_len = strlen(plaintext);
    uint8_t nonce[FC_NONCE_BYTES];
    fc_random_nonce(nonce);
    uint8_t ct[64], pt2[64], tag[FC_TAG_BYTES];
    fc_encrypt(shared_ab, nonce, NULL, 0,
               (const uint8_t *)plaintext, ct, pt_len, tag);
    int ok = fc_decrypt(shared_ab, nonce, NULL, 0,
                        ct, pt2, pt_len, tag);
    if (ok == 0 && memcmp(plaintext, pt2, pt_len) == 0) {
        printf("PASS\n");
    } else {
        printf("FAIL\n"); pass = 0;
    }

    /* Test 3: tag verification rejects tampered ciphertext */
    printf("Test 3: tamper detection ... ");
    uint8_t ct_tampered[64];
    memcpy(ct_tampered, ct, pt_len);
    ct_tampered[0] ^= 0xFF;
    uint8_t pt_bad[64];
    int bad = fc_decrypt(shared_ab, nonce, NULL, 0,
                         ct_tampered, pt_bad, pt_len, tag);
    if (bad == -1) {
        printf("PASS\n");
    } else {
        printf("FAIL (tamper not detected)\n"); pass = 0;
    }

    /* Test 4: wrong key rejected */
    printf("Test 4: wrong key rejected ... ");
    FcKeypair eve;
    fc_keygen(&eve);
    uint8_t shared_ev[FC_SHARED_BYTES];
    fc_shared_secret(eve.priv, bob.pub, shared_ev);
    uint8_t pt_wrong[64];
    int wrong = fc_decrypt(shared_ev, nonce, NULL, 0,
                           ct, pt_wrong, pt_len, tag);
    if (wrong == -1) {
        printf("PASS\n");
    } else {
        printf("FAIL (wrong key accepted)\n"); pass = 0;
    }

    /* Test 5: deterministic — same inputs → same ciphertext */
    printf("Test 5: deterministic encryption ... ");
    uint8_t ct_a[64], ct_b[64], tag_a[FC_TAG_BYTES], tag_b[FC_TAG_BYTES];
    fc_encrypt(shared_ab, nonce, NULL, 0,
               (const uint8_t *)plaintext, ct_a, pt_len, tag_a);
    fc_encrypt(shared_ab, nonce, NULL, 0,
               (const uint8_t *)plaintext, ct_b, pt_len, tag_b);
    if (memcmp(ct_a, ct_b, pt_len) == 0 &&
        memcmp(tag_a, tag_b, FC_TAG_BYTES) == 0) {
        printf("PASS\n");
    } else {
        printf("FAIL\n"); pass = 0;
    }

    /* Test 6: different nonces → different ciphertext */
    printf("Test 6: nonce uniqueness ... ");
    uint8_t nonce2[FC_NONCE_BYTES];
    fc_random_nonce(nonce2);
    uint8_t ct_n2[64], tag_n2[FC_TAG_BYTES];
    fc_encrypt(shared_ab, nonce2, NULL, 0,
               (const uint8_t *)plaintext, ct_n2, pt_len, tag_n2);
    if (memcmp(ct_a, ct_n2, pt_len) != 0) {
        printf("PASS\n");
    } else {
        printf("FAIL (same ciphertext for different nonces)\n"); pass = 0;
    }

    printf("\n%s\n", pass ? "All tests passed." : "SOME TESTS FAILED.");
    return pass ? 0 : 1;
}

/* ── main ─────────────────────────────────────────────────────────────────── */

static void usage(const char *prog) {
    fprintf(stderr,
        "FractalCipher — sponge-based authenticated stream cipher\n\n"
        "Usage:\n"
        "  %s keygen  <privkey_out> <pubkey_out>\n"
        "  %s shared  <my_privkey> <peer_pubkey> <shared_out>\n"
        "  %s encrypt <shared_file> <plaintext_in> <ciphertext_out>\n"
        "  %s decrypt <shared_file> <ciphertext_in> <plaintext_out>\n"
        "  %s test\n\n"
        "Workflow:\n"
        "  Alice: fractal_cipher keygen alice.priv alice.pub\n"
        "  Bob:   fractal_cipher keygen bob.priv   bob.pub\n"
        "  Alice: fractal_cipher shared alice.priv bob.pub shared.key\n"
        "  Bob:   fractal_cipher shared bob.priv alice.pub shared.key\n"
        "         (both produce identical shared.key)\n"
        "  Alice: fractal_cipher encrypt shared.key message.txt message.enc\n"
        "  Bob:   fractal_cipher decrypt shared.key message.enc message.txt\n",
        prog, prog, prog, prog, prog);
}

int main(int argc, char *argv[]) {
    if (argc < 2) { usage(argv[0]); return 1; }

    const char *cmd = argv[1];
    if (strcmp(cmd, "keygen") == 0 && argc == 4)
        return cmd_keygen(argv[2], argv[3]);
    if (strcmp(cmd, "shared") == 0 && argc == 5)
        return cmd_shared(argv[2], argv[3], argv[4]);
    if (strcmp(cmd, "encrypt") == 0 && argc == 5)
        return cmd_encrypt(argv[2], argv[3], argv[4]);
    if (strcmp(cmd, "decrypt") == 0 && argc == 5)
        return cmd_decrypt(argv[2], argv[3], argv[4]);
    if (strcmp(cmd, "test") == 0)
        return cmd_test();

    usage(argv[0]);
    return 1;
}

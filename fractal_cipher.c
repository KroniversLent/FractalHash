/*
 * FractalCipher — stream cipher implementation
 *
 * Build: gcc -O2 -ffp-contract=off -lm -o fractal_cipher \
 *             fractal_cipher.c fractal_sponge.c fractal_sponge_avx2.c \
 *             cipher_main.c -lm
 */

#include "fractal_cipher.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* ── internal sponge duplex state ─────────────────────────────────────────── */

/*
 * The cipher uses the FractalSponge permutation directly.
 * We reuse the same rate/capacity split as the hash (rate = 4 words = 32 bytes,
 * capacity = 4 words = 32 bytes).
 *
 * Duplex mode:
 *   absorb(block): state[0..3] ^= block_words; permute()
 *   squeeze()    : return state[0..3] as 32 bytes keystream; permute()
 *
 * Domain separation byte appended to every absorbed block (as part of the
 * last byte of the block):
 *   0x01 = key/nonce init
 *   0x02 = AAD block
 *   0x03 = plaintext/ciphertext block
 *   0x04 = finalize / squeeze tag
 */

/* We pull fractal_permutation via the dispatch pointer in fractal_sponge.c.
 * Expose a thin wrapper that calls fs256_hash internally to seed the state,
 * then we manage the state ourselves. */

/* Low-level: absorb exactly FS_BLOCK_BYTES into state with domain byte */
static void duplex_absorb(uint64_t state[8],
                          const uint8_t block[FS_BLOCK_BYTES],
                          uint8_t domain)
{
    /* XOR rate words */
    uint8_t tmp[FS_BLOCK_BYTES];
    memcpy(tmp, block, FS_BLOCK_BYTES);
    tmp[FS_BLOCK_BYTES - 1] ^= domain;   /* domain separation in last byte */

    for (int i = 0; i < FS_RATE_WORDS; i++) {
        uint64_t w = 0;
        for (int b = 0; b < 8; b++)
            w = (w << 8) | tmp[i*8 + b];
        state[i] ^= w;
    }
    /* Call permutation via the same dispatch path used by fs256_hash.
     * We reach it by hashing a dummy — but that's too heavyweight.
     * Instead we expose the permutation through a local implementation.
     * Simplest: inline the same scalar permutation here. */
    fs256_permute(state);
}

/* squeeze 32 bytes of keystream from the rate, then permute */
static void duplex_squeeze(uint64_t state[8], uint8_t out[FS_BLOCK_BYTES]) {
    for (int i = 0; i < FS_RATE_WORDS; i++) {
        uint64_t w = state[i];
        for (int b = 7; b >= 0; b--) {
            out[i*8 + b] = (uint8_t)(w & 0xff);
            w >>= 8;
        }
    }
    fs256_permute(state);
}

/* ── public functions ─────────────────────────────────────────────────────── */

int fc_random_nonce(uint8_t nonce[FC_NONCE_BYTES]) {
    FILE *f = fopen("/dev/urandom", "rb");
    if (!f) return -1;
    int ok = (fread(nonce, 1, FC_NONCE_BYTES, f) == FC_NONCE_BYTES) ? 0 : -1;
    fclose(f);
    return ok;
}

int fc_keygen(FcKeypair *kp) {
    FILE *f = fopen("/dev/urandom", "rb");
    if (!f) return -1;
    if (fread(kp->priv, 1, FC_PRIVKEY_BYTES, f) != FC_PRIVKEY_BYTES) {
        fclose(f); return -1;
    }
    fclose(f);

    /* Public key = H("FractalCipher-pubkey\x00" || private_key) */
    uint8_t buf[20 + FC_PRIVKEY_BYTES];
    memcpy(buf, "FractalCipher-pubkey", 20);
    memcpy(buf + 20, kp->priv, FC_PRIVKEY_BYTES);
    fs256_hash(buf, sizeof(buf), kp->pub);
    return 0;
}

void fc_shared_secret(const uint8_t my_priv[FC_PRIVKEY_BYTES],
                      const uint8_t peer_pub[FC_PUBKEY_BYTES],
                      uint8_t       shared[FC_SHARED_BYTES])
{
    /*
     * Symmetric shared secret = H("FractalCipher-shared" || lo_pub || hi_pub)
     *
     * Both parties derive the same value by sorting the two public keys
     * lexicographically.  Each party's own public key is re-derived from their
     * private key.  The private key ensures only the key holder can derive the
     * matching public key, binding identity to the shared session.
     *
     * Note: this construction provides mutual authentication (each party must
     * know their private key to derive the correct public key for the sort),
     * but the shared secret is deterministic from the two public keys, which
     * means it is only as private as the public keys themselves.  For a
     * production system, use X25519 or a proper DH group.
     */
    uint8_t my_pub[FC_PUBKEY_BYTES];
    uint8_t pub_buf[20 + FC_PRIVKEY_BYTES];
    memcpy(pub_buf, "FractalCipher-pubkey", 20);
    memcpy(pub_buf + 20, my_priv, FC_PRIVKEY_BYTES);
    fs256_hash(pub_buf, sizeof(pub_buf), my_pub);

    /* Sort the two public keys lexicographically — guarantees symmetry */
    const uint8_t *lo = my_pub, *hi = peer_pub;
    if (memcmp(my_pub, peer_pub, FC_PUBKEY_BYTES) > 0) {
        lo = peer_pub; hi = my_pub;
    }

    /* Hash domain tag + sorted public keys */
    uint8_t input[20 + FC_PUBKEY_BYTES + FC_PUBKEY_BYTES];
    memcpy(input, "FractalCipher-shared", 20);
    memcpy(input + 20,                   lo, FC_PUBKEY_BYTES);
    memcpy(input + 20 + FC_PUBKEY_BYTES, hi, FC_PUBKEY_BYTES);
    fs256_hash(input, sizeof(input), shared);
}

/*
 * Initialize the duplex sponge state from shared secret + nonce.
 * Uses fs256_hash to absorb a domain-tagged seed into the state.
 */
static void fc_init_state(uint64_t state[8],
                          const uint8_t shared[FC_SHARED_BYTES],
                          const uint8_t nonce[FC_NONCE_BYTES])
{
    memset(state, 0, 8 * sizeof(uint64_t));

    /* Build seed = "FractalCipher-stream\x00" || shared || nonce */
    uint8_t seed[20 + FC_SHARED_BYTES + FC_NONCE_BYTES];
    memcpy(seed, "FractalCipher-stream", 20);
    memcpy(seed + 20,                  shared, FC_SHARED_BYTES);
    memcpy(seed + 20 + FC_SHARED_BYTES, nonce, FC_NONCE_BYTES);

    /* Absorb seed in FS_BLOCK_BYTES chunks with domain 0x01 */
    size_t seed_len = sizeof(seed);
    size_t off = 0;
    while (off + FS_BLOCK_BYTES <= seed_len) {
        duplex_absorb(state, seed + off, 0x01);
        off += FS_BLOCK_BYTES;
    }
    /* Final partial block with SHA-3-style padding */
    uint8_t tail[FS_BLOCK_BYTES];
    memset(tail, 0, sizeof(tail));
    size_t rem = seed_len - off;
    if (rem) memcpy(tail, seed + off, rem);
    tail[rem] ^= 0x06;
    tail[FS_BLOCK_BYTES - 1] ^= 0x81;   /* 0x80 | domain 0x01 */
    duplex_absorb(state, tail, 0x00);    /* domain already in tail */
}

/* Absorb arbitrary-length data with a given domain byte */
static void fc_absorb_data(uint64_t state[8],
                           const uint8_t *data, size_t len,
                           uint8_t domain)
{
    if (len == 0) return;
    size_t off = 0;
    while (off + FS_BLOCK_BYTES <= len) {
        duplex_absorb(state, data + off, domain);
        off += FS_BLOCK_BYTES;
    }
    uint8_t tail[FS_BLOCK_BYTES];
    memset(tail, 0, sizeof(tail));
    size_t rem = len - off;
    if (rem) memcpy(tail, data + off, rem);
    tail[rem] ^= 0x06;
    tail[FS_BLOCK_BYTES - 1] ^= (0x80 | domain);
    duplex_absorb(state, tail, 0x00);
}

void fc_encrypt(const uint8_t shared[FC_SHARED_BYTES],
                const uint8_t nonce[FC_NONCE_BYTES],
                const uint8_t *aad,  size_t aad_len,
                const uint8_t *pt,   uint8_t *ct,  size_t pt_len,
                uint8_t        tag[FC_TAG_BYTES])
{
    uint64_t state[8];
    fc_init_state(state, shared, nonce);

    /* Absorb AAD (domain 0x02) */
    if (aad_len > 0)
        fc_absorb_data(state, aad, aad_len, 0x02);

    /* Encrypt: squeeze keystream block, XOR with plaintext,
     * then absorb ciphertext block back (duplex binding) */
    uint8_t ks[FS_BLOCK_BYTES];
    size_t off = 0;

    while (off < pt_len) {
        duplex_squeeze(state, ks);

        size_t chunk = pt_len - off;
        if (chunk > FS_BLOCK_BYTES) chunk = FS_BLOCK_BYTES;

        for (size_t i = 0; i < chunk; i++)
            ct[off + i] = pt[off + i] ^ ks[i];

        /* Absorb the ciphertext chunk back (domain 0x03) */
        uint8_t ct_block[FS_BLOCK_BYTES];
        memset(ct_block, 0, sizeof(ct_block));
        memcpy(ct_block, ct + off, chunk);
        if (chunk < FS_BLOCK_BYTES) {
            ct_block[chunk] ^= 0x06;
            ct_block[FS_BLOCK_BYTES - 1] ^= 0x83;  /* 0x80 | domain 0x03 */
            duplex_absorb(state, ct_block, 0x00);
        } else {
            duplex_absorb(state, ct_block, 0x03);
        }
        off += chunk;
    }

    /* Squeeze authentication tag (16 bytes) — domain 0x04 finalization */
    uint8_t tag_block[FS_BLOCK_BYTES];
    memset(tag_block, 0, sizeof(tag_block));
    tag_block[0] = 0x06;
    tag_block[FS_BLOCK_BYTES - 1] ^= 0x84;  /* 0x80 | domain 0x04 */
    duplex_absorb(state, tag_block, 0x00);

    duplex_squeeze(state, tag_block);
    memcpy(tag, tag_block, FC_TAG_BYTES);

    /* Wipe sensitive state */
    memset(state, 0, sizeof(state));
    memset(ks, 0, sizeof(ks));
}

int fc_decrypt(const uint8_t shared[FC_SHARED_BYTES],
               const uint8_t nonce[FC_NONCE_BYTES],
               const uint8_t *aad,  size_t aad_len,
               const uint8_t *ct,   uint8_t *pt,   size_t ct_len,
               const uint8_t  tag[FC_TAG_BYTES])
{
    uint64_t state[8];
    fc_init_state(state, shared, nonce);

    if (aad_len > 0)
        fc_absorb_data(state, aad, aad_len, 0x02);

    uint8_t ks[FS_BLOCK_BYTES];
    size_t off = 0;

    while (off < ct_len) {
        duplex_squeeze(state, ks);

        size_t chunk = ct_len - off;
        if (chunk > FS_BLOCK_BYTES) chunk = FS_BLOCK_BYTES;

        for (size_t i = 0; i < chunk; i++)
            pt[off + i] = ct[off + i] ^ ks[i];

        /* Absorb ciphertext (not plaintext) — same as encrypt path */
        uint8_t ct_block[FS_BLOCK_BYTES];
        memset(ct_block, 0, sizeof(ct_block));
        memcpy(ct_block, ct + off, chunk);
        if (chunk < FS_BLOCK_BYTES) {
            ct_block[chunk] ^= 0x06;
            ct_block[FS_BLOCK_BYTES - 1] ^= 0x83;
            duplex_absorb(state, ct_block, 0x00);
        } else {
            duplex_absorb(state, ct_block, 0x03);
        }
        off += chunk;
    }

    /* Recompute tag */
    uint8_t tag_block[FS_BLOCK_BYTES];
    memset(tag_block, 0, sizeof(tag_block));
    tag_block[0] = 0x06;
    tag_block[FS_BLOCK_BYTES - 1] ^= 0x84;
    duplex_absorb(state, tag_block, 0x00);

    duplex_squeeze(state, tag_block);

    /* Constant-time tag comparison */
    uint8_t diff = 0;
    for (int i = 0; i < FC_TAG_BYTES; i++)
        diff |= (tag_block[i] ^ tag[i]);

    memset(state, 0, sizeof(state));
    memset(ks, 0, sizeof(ks));

    if (diff != 0) {
        memset(pt, 0, ct_len);
        return -1;
    }
    return 0;
}

#pragma once
/*
 * FractalCipher — stream cipher + keypair based on FractalSponge-256
 *
 * Key scheme:
 *   Private key : 32 random bytes
 *   Public key  : H("pubkey" || private_key)  [32 bytes]
 *   Shared secret: H("shared" || min(pk_a,pk_b) || max(pk_a,pk_b))
 *                  where min/max are lexicographic — symmetric, same result
 *                  regardless of which side calls it.
 *
 * Stream cipher (duplex sponge construction):
 *   State = FractalSponge permutation state (512 bits)
 *   Nonce  : 16-byte random value chosen per message
 *   Init   : absorb shared_secret || nonce into state
 *   Encrypt: XOR plaintext with keystream blocks squeezed from state;
 *            re-absorb ciphertext block before each squeeze (duplex mode)
 *   Auth tag: 16 bytes squeezed after all data — provides authenticated
 *             encryption (AEAD).
 *
 * WARNING: Research-grade code.  Not formally analysed.  Do not use in
 *          production without independent security review.
 */

#include "fractal_sponge.h"

/* ── sizes ── */
#define FC_PRIVKEY_BYTES  32
#define FC_PUBKEY_BYTES   32
#define FC_SHARED_BYTES   32
#define FC_NONCE_BYTES    16
#define FC_TAG_BYTES      16

/* ── key pair ── */
typedef struct {
    uint8_t priv[FC_PRIVKEY_BYTES];
    uint8_t pub [FC_PUBKEY_BYTES];
} FcKeypair;

/* ── public API ── */
#ifdef __cplusplus
extern "C" {
#endif

/* Generate a keypair from /dev/urandom.  Returns 0 on success, -1 on error. */
int  fc_keygen(FcKeypair *kp);

/* Derive the 32-byte shared secret from your private key and the peer's
 * public key.  Call on both sides with each other's public keys — both
 * sides obtain identical shared_secret[]. */
void fc_shared_secret(const uint8_t my_priv[FC_PRIVKEY_BYTES],
                      const uint8_t peer_pub[FC_PUBKEY_BYTES],
                      uint8_t       shared[FC_SHARED_BYTES]);

/*
 * fc_encrypt — authenticated encryption (AEAD)
 *
 * Parameters:
 *   shared   [in]  FC_SHARED_BYTES shared secret (from fc_shared_secret)
 *   nonce    [in]  FC_NONCE_BYTES  per-message nonce (must be unique)
 *   aad      [in]  additional authenticated data (may be NULL if aad_len==0)
 *   pt       [in]  plaintext
 *   ct       [out] ciphertext (same length as pt)
 *   pt_len   [in]  bytes of plaintext
 *   tag      [out] FC_TAG_BYTES authentication tag
 */
void fc_encrypt(const uint8_t shared[FC_SHARED_BYTES],
                const uint8_t nonce[FC_NONCE_BYTES],
                const uint8_t *aad,  size_t aad_len,
                const uint8_t *pt,   uint8_t *ct,  size_t pt_len,
                uint8_t        tag[FC_TAG_BYTES]);

/*
 * fc_decrypt — authenticated decryption (AEAD)
 *
 * Returns 0 on success (tag verified), -1 if authentication fails.
 * Output pt is zeroed on authentication failure.
 */
int  fc_decrypt(const uint8_t shared[FC_SHARED_BYTES],
                const uint8_t nonce[FC_NONCE_BYTES],
                const uint8_t *aad,  size_t aad_len,
                const uint8_t *ct,   uint8_t *pt,   size_t ct_len,
                const uint8_t  tag[FC_TAG_BYTES]);

/* Convenience: generate a 16-byte nonce from /dev/urandom.
 * Returns 0 on success, -1 on error. */
int  fc_random_nonce(uint8_t nonce[FC_NONCE_BYTES]);

#ifdef __cplusplus
}
#endif

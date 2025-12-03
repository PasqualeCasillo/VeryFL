# chainfl/crypto_bbs.py
"""
Cryptographic primitives for VeryFL decentralized FL with BBS+ group signatures.
Provides:
- AES-GCM encryption/decryption
- BBS+ group signatures (sign/verify)
- Helper functions for encoding
"""

import os
import base64
import hashlib
import json
from typing import List, Tuple, Dict, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey

from ursa_bbs_signatures import (
    SignRequest,
    VerifyRequest,
    BlsKeyPair,
    sign as bbs_sign,
    verify as bbs_verify,
)

import logging
logger = logging.getLogger(__name__)


# ==================== ENCODING UTILITIES ====================

def sha256(b: bytes) -> bytes:
    """Compute SHA-256 hash"""
    return hashlib.sha256(b).digest()


def b64e(b: bytes) -> str:
    """Base64url encode (no padding)"""
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def b64d(s: str) -> bytes:
    """Base64url decode (add padding)"""
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


# ==================== AES-GCM ENCRYPTION ====================

def aesgcm_encrypt(key: bytes, plaintext: bytes, aad: Optional[bytes] = None) -> Dict[str, str]:
    """
    Encrypt plaintext with AES-256-GCM
    
    Args:
        key: 32-byte encryption key
        plaintext: data to encrypt
        aad: optional additional authenticated data
    
    Returns:
        Dict with 'nonce' and 'ct' (both base64url encoded)
    """
    if len(key) != 32:
        raise ValueError("AES-GCM key must be 32 bytes")
    
    aes = AESGCM(key)
    nonce = os.urandom(12)  # 96-bit nonce for GCM
    ciphertext = aes.encrypt(nonce, plaintext, aad)
    
    return {
        "nonce": b64e(nonce),
        "ct": b64e(ciphertext)
    }


def aesgcm_decrypt(key: bytes, enc: Dict[str, str], aad: Optional[bytes] = None) -> bytes:
    """
    Decrypt AES-256-GCM ciphertext
    
    Args:
        key: 32-byte decryption key
        enc: dict with 'nonce' and 'ct' (base64url encoded)
        aad: optional additional authenticated data (must match encryption)
    
    Returns:
        Decrypted plaintext bytes
    """
    if len(key) != 32:
        raise ValueError("AES-GCM key must be 32 bytes")
    
    aes = AESGCM(key)
    nonce = b64d(enc["nonce"])
    ciphertext = b64d(enc["ct"])
    
    try:
        plaintext = aes.decrypt(nonce, ciphertext, aad)
        return plaintext
    except Exception as e:
        raise ValueError(f"Decryption failed: {e}")


# ==================== BBS+ GROUP SIGNATURES ====================

def bbs_generate_keypair(seed: bytes = None) -> Tuple[bytes, bytes]:
    """
    Generate a BLS12-381 keypair for BBS+ signatures
    
    Args:
        seed: optional 32-byte seed (if None, uses random)
    
    Returns:
        (public_key, secret_key) tuple
    """
    if seed is None:
        seed = os.urandom(32)
    elif len(seed) != 32:
        raise ValueError("Seed must be 32 bytes")
    
    kp = BlsKeyPair.generate_g2(seed)
    return kp.public_key, kp.secret_key


def bbs_sign_messages(messages: List[bytes], secret_key: bytes) -> bytes:
    """
    Sign multiple messages with BBS+ (group signature)
    
    Args:
        messages: list of byte messages to sign
        secret_key: signer's BBS+ secret key
    
    Returns:
        BBS+ signature bytes
    """
    # Convert bytes to strings (ursa-bbs-signatures requires strings)
    msgs_str = []
    for m in messages:
        if isinstance(m, bytes):
            msgs_str.append(m.hex())  # Convert bytes to hex string
        else:
            msgs_str.append(str(m))
    
    try:
        key_pair = BlsKeyPair.from_secret_key(secret_key)
        req = SignRequest(key_pair=key_pair, messages=msgs_str)
        signature = bbs_sign(req)
        return signature
    except Exception as e:
        logger.error(f"BBS+ signing failed: {e}")
        raise


def bbs_verify_messages(messages: List[bytes], signature: bytes, public_key: bytes) -> bool:
    """
    Verify BBS+ signature against a single public key
    
    Args:
        messages: list of messages that were signed
        signature: BBS+ signature to verify
        public_key: signer's public key
    
    Returns:
        True if signature is valid, False otherwise
    """
    # Convert bytes to strings
    msgs_str = []
    for m in messages:
        if isinstance(m, bytes):
            msgs_str.append(m.hex())
        else:
            msgs_str.append(str(m))
    
    try:
        kp = BlsKeyPair(public_key=public_key)
        req = VerifyRequest(key_pair=kp, signature=signature, messages=msgs_str)
        result = bbs_verify(req)
        return result
    except Exception as e:
        logger.debug(f"BBS+ verification failed: {e}")
        return False


def group_verify_any(pubkeys: List[bytes], messages: List[bytes], signature: bytes) -> bool:
    """
    Verify that signature is valid for AT LEAST ONE public key in the group.
    This provides anonymous authentication: proves membership without revealing identity.
    
    Args:
        pubkeys: list of group member public keys
        messages: messages that were signed
        signature: BBS+ signature to verify
    
    Returns:
        True if signature is valid for any group member, False otherwise
    """
    for pk in pubkeys:
        try:
            if bbs_verify_messages(messages, signature, pk):
                logger.debug(" Group signature verified (member authenticated anonymously)")
                return True
        except Exception:
            continue  # Try next key
    
    logger.warning("Group signature verification FAILED - not from any authorized member")
    return False


# ==================== X25519 KEY EXCHANGE (OPTIONAL - for KRS) ====================

def x25519_generate() -> Tuple[str, str]:
    """
    Generate ephemeral X25519 keypair (for HPKE key wrapping)
    
    Returns:
        (secret_key_b64, public_key_b64) tuple
    """
    sk = X25519PrivateKey.generate()
    pk = sk.public_key()
    
    sk_bytes = sk.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption()
    )
    pk_bytes = pk.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )
    
    return b64e(sk_bytes), b64e(pk_bytes)


def hpke_seal(pk_recipient_b64: str, plaintext: bytes, info: bytes = b"hpke-krs") -> Dict[str, str]:
    """
    HPKE-like encryption: encrypt plaintext to recipient's public key
    (Simplified version using X25519 + HKDF + AES-GCM)
    
    Args:
        pk_recipient_b64: recipient's X25519 public key (base64url)
        plaintext: data to encrypt
        info: optional context string
    
    Returns:
        Dict with 'pk_eph', 'nonce', 'ct' (all base64url encoded)
    """
    pk_recipient = X25519PublicKey.from_public_bytes(b64d(pk_recipient_b64))
    sk_eph = X25519PrivateKey.generate()
    pk_eph = sk_eph.public_key()
    
    # ECDH
    shared = sk_eph.exchange(pk_recipient)
    
    # KDF: derive encryption key
    kek = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=info
    ).derive(shared)
    
    # Encrypt with derived key
    pk_eph_bytes = pk_eph.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )
    aad = pk_eph_bytes
    wrap = aesgcm_encrypt(kek, plaintext, aad=aad)
    wrap.update({"pk_eph": b64e(pk_eph_bytes)})
    
    return wrap


def hpke_open(sk_recipient_b64: str, wrap: Dict[str, str], info: bytes = b"hpke-krs") -> bytes:
    """
    HPKE-like decryption: decrypt using recipient's secret key
    
    Args:
        sk_recipient_b64: recipient's X25519 secret key (base64url)
        wrap: dict with 'pk_eph', 'nonce', 'ct'
        info: context string (must match seal)
    
    Returns:
        Decrypted plaintext bytes
    """
    sk = X25519PrivateKey.from_private_bytes(b64d(sk_recipient_b64))
    pk_eph = X25519PublicKey.from_public_bytes(b64d(wrap["pk_eph"]))
    
    # ECDH
    shared = sk.exchange(pk_eph)
    
    # KDF: derive decryption key
    kek = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=info
    ).derive(shared)
    
    # Decrypt
    plaintext = aesgcm_decrypt(kek, {"nonce": wrap["nonce"], "ct": wrap["ct"]}, aad=b64d(wrap["pk_eph"]))
    return plaintext


# ==================== TESTING ====================

if __name__ == "__main__":
    print("=== Testing crypto_bbs.py ===\n")
    
    # Test 1: AES-GCM encryption
    print("1. Testing AES-GCM encryption...")
    key = os.urandom(32)
    plaintext = b"Hello VeryFL Decentralized!"
    encrypted = aesgcm_encrypt(key, plaintext)
    decrypted = aesgcm_decrypt(key, encrypted)
    assert decrypted == plaintext
    print(f"    Encrypted/Decrypted successfully")
    print(f"   Ciphertext preview: {encrypted['ct'][:40]}...\n")
    
    # Test 2: BBS+ Group Signatures
    print("2. Testing BBS+ group signatures...")
    
    # Generate 3 group member keys
    pubkeys = []
    seckeys = []
    for i in range(3):
        pk, sk = bbs_generate_keypair()
        pubkeys.append(pk)
        seckeys.append(sk)
    
    # Member 0 signs a message
    message = b"Model weights for round 42"
    msg_hash = sha256(message)
    signature = bbs_sign_messages([msg_hash], seckeys[0])
    
    print(f"   Generated {len(pubkeys)} group member keys")
    print(f"   Signature size: {len(signature)} bytes")
    
    # Verify with group (should succeed)
    is_valid = group_verify_any(pubkeys, [msg_hash], signature)
    assert is_valid
    print(f"    Group signature verified (anonymous authentication)")
    
    # Try verifying with wrong message (should fail)
    wrong_msg = sha256(b"Wrong message")
    is_valid_wrong = group_verify_any(pubkeys, [wrong_msg], signature)
    assert not is_valid_wrong
    print(f"    Invalid signature correctly rejected\n")
    
    # Test 3: HPKE key wrapping (optional)
    print("3. Testing HPKE key wrapping...")
    sk_rec, pk_rec = x25519_generate()
    secret = os.urandom(32)
    wrapped = hpke_seal(pk_rec, secret)
    unwrapped = hpke_open(sk_rec, wrapped)
    assert unwrapped == secret
    print(f"    HPKE wrap/unwrap successful\n")
    
    print("=== All tests passed! ===")
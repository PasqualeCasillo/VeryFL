# test_hpke_integration.py
"""
Test HPKE integration nel KRS
"""
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from chainfl.crypto_bbs import (
    bbs_generate_keypair,
    bbs_sign_messages,
    sha256,
    x25519_generate,
    hpke_seal,
    hpke_open,
    b64e,
    b64d
)
from chainfl.krs import KeyReleaseService

def test_hpke_krs_flow():
    print("=" * 60)
    print("TEST: HPKE-enabled KRS Flow")
    print("=" * 60)
    
    # 1. Setup gruppo con 3 membri
    print("\n1. Setting up group with 3 members...")
    group_public_keys = []
    member_secret_keys = []
    
    for i in range(3):
        pk, sk = bbs_generate_keypair()
        group_public_keys.append(pk)
        member_secret_keys.append(sk)
        print(f"   Member {i}: keypair generated")
    
    # 2. Inizializza KRS
    print("\n2. Initializing KRS...")
    krs = KeyReleaseService(group_public_keys)
    
    # 3. Publisher registra chiave
    print("\n3. Publisher registers encryption key...")
    cid_manifest = "QmTest123"
    Kc_original = os.urandom(32)
    Kc_b64 = b64e(Kc_original)
    krs.register_key(cid_manifest, Kc_b64)
    print(f"   Registered key for {cid_manifest}")
    
    # 4. Consumer (Member 0) richiede chiave
    print("\n4. Consumer requests key with HPKE...")
    
    # 4a. Genera chiave effimera
    sk_eph_b64, pk_eph_b64 = x25519_generate()
    print(f"   Generated ephemeral keypair")
    print(f"   pk_eph: {pk_eph_b64[:30]}...")
    
    # 4b. Genera nonce
    nonce = os.urandom(16).hex()
    
    # 4c. Firma richiesta
    message = sha256(f"{cid_manifest}||{nonce}||{pk_eph_b64}".encode())
    signature = bbs_sign_messages([message], member_secret_keys[0])
    print(f"   Signed request with BBS+")
    
    # 4d. Richiedi chiave
    wrapped = krs.request_key(cid_manifest, nonce, signature, pk_eph_b64)
    
    if not wrapped:
        print("    FAILED: KRS denied request")
        return False
    
    print(f"    Received wrapped key")
    print(f"   Wrapped format: {wrapped.keys()}")
    
    # 5. Consumer decifra chiave
    print("\n5. Consumer unwraps key...")
    try:
        Kc_recovered_bytes = hpke_open(sk_eph_b64, wrapped, info=b"veryfl-krs-v1")
        Kc_recovered_b64 = b64e(Kc_recovered_bytes)
        
        print(f"    Key unwrapped successfully")
        
        # 6. Verifica correttezza
        if Kc_recovered_b64 == Kc_b64:
            print(f"    Key matches original!")
            print(f"\n{'='*60}")
            print("TEST PASSED ")
            print(f"{'='*60}")
            return True
        else:
            print(f"    Key mismatch!")
            print(f"   Original:  {Kc_b64[:30]}...")
            print(f"   Recovered: {Kc_recovered_b64[:30]}...")
            return False
            
    except Exception as e:
        print(f"    Unwrap failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rate_limiting():
    print("\n" + "=" * 60)
    print("TEST: Rate Limiting")
    print("=" * 60)
    
    # Setup
    pk, sk = bbs_generate_keypair()
    krs = KeyReleaseService([pk])
    cid = "QmRateTest"
    krs.register_key(cid, b64e(os.urandom(32)))
    
    sk_eph_b64, pk_eph_b64 = x25519_generate()
    
    # Fai 12 richieste (limite è 10)
    print("\nMaking 12 requests (limit is 10)...")
    success_count = 0
    denied_count = 0
    
    for i in range(12):
        nonce = os.urandom(16).hex()
        message = sha256(f"{cid}||{nonce}||{pk_eph_b64}".encode())
        signature = bbs_sign_messages([message], sk)
        
        wrapped = krs.request_key(cid, nonce, signature, pk_eph_b64)
        
        if wrapped:
            success_count += 1
            print(f"   Request {i+1}:  Granted")
        else:
            denied_count += 1
            print(f"   Request {i+1}:  Denied (rate limit)")
    
    print(f"\nResults: {success_count} granted, {denied_count} denied")
    
    if success_count == 10 and denied_count == 2:
        print(" Rate limiting works correctly!")
        return True
    else:
        print(f" Expected 10 granted, 2 denied")
        return False

if __name__ == "__main__":
    test1 = test_hpke_krs_flow()
    test2 = test_rate_limiting()
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"HPKE Flow: {' PASS' if test1 else ' FAIL'}")
    print(f"Rate Limiting: {' PASS' if test2 else ' FAIL'}")
    print("=" * 60)
# test_ipfs_crypto.py

import os
import sys
sys.path.insert(0, '.')

from chainfl.ipfs_client import IPFSClient
from chainfl.crypto_bbs import bbs_generate_keypair
import torch

def test_ipfs_crypto():
    print("Testing IPFS + Crypto integration...")
    
    # Setup
    ipfs_client = IPFSClient()
    
    # Generate group keys
    pubkeys = []
    seckeys = []
    for i in range(3):
        pk, sk = bbs_generate_keypair()
        pubkeys.append(pk)
        seckeys.append(sk)
    
    ipfs_client.set_group_keys(pubkeys)
    ipfs_client.enable_encryption(True)
    
    # Create dummy model
    dummy_model = {
        'layer1.weight': torch.randn(10, 5),
        'layer1.bias': torch.randn(10)
    }
    
    metadata = {
        'node_id': '1',
        'round': 42
    }
    
    # Upload
    print("\n1. Testing upload_model_secured...")
    cid_manifest, key_data = ipfs_client.upload_model_secured(
        dummy_model, 
        metadata, 
        seckeys[0]
    )
    
    if cid_manifest and key_data:
        print(f"   Manifest CID: {cid_manifest}")
        print(f"   Cipher CID: {key_data.get('cid_cipher', 'N/A')}")
        print(f"   Encrypted: {key_data.get('encrypted', False)}")
        
        # Download
        print("\n2. Testing download_model_secured...")
        downloaded = ipfs_client.download_model_secured(
            cid_manifest, 
            key_data['Kc']
        )
        
        if downloaded:
            print("   Model downloaded and verified successfully")
            
            # Verify content
            if torch.allclose(downloaded['layer1.weight'], dummy_model['layer1.weight']):
                print("   Content verification: PASSED")
            else:
                print("   Content verification: FAILED")
        else:
            print("   Download failed")
    else:
        print("   Upload failed")

if __name__ == "__main__":
    test_ipfs_crypto()
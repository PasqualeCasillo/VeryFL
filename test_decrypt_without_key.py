# test_decrypt_without_key.py

import pickle
import json
import ipfshttpclient

# Connetti IPFS
client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001')

# Scarica ciphertext
cid_cipher = "QmZ9Y8cmsyDnXyp3pYx3TtdBmQA1m4ku1p9dSVtBERRDiy"
ciphertext_json = client.cat(cid_cipher)
ciphertext_data = json.loads(ciphertext_json)

print("Ciphertext structure:")
print(f"  Keys: {ciphertext_data.keys()}")
print(f"  Nonce length: {len(ciphertext_data['nonce'])} chars")
print(f"  CT length: {len(ciphertext_data['ct'])} chars")
print()

# Prova a decodificare direttamente (DEVE FALLIRE)
from chainfl.crypto_bbs import b64d

ct_bytes = b64d(ciphertext_data['ct'])
print(f"Raw ciphertext bytes: {ct_bytes[:50]}...")
print()

try:
    # Tentativo di deserializzare senza decifrare
    model = pickle.loads(ct_bytes)
    print("ERROR: Model deserialized without decryption!")
except Exception as e:
    print(f"SUCCESS: Cannot deserialize without decryption")
    print(f"  Error: {type(e).__name__}: {str(e)[:100]}")
    
    
    
# Step 1: Estrai CID dal log
# Ciphertext uploaded to IPFS: QmZ9Y8cmsyDnXyp3pYx3TtdBmQA1m4ku1p9dSVtBERRDiy
# Signed manifest uploaded to IPFS: QmXjHJDGRzsj8i3aRBvKhYNbUD1F2hUxarAZWHAST8TngL
# Scarica il ciphertext
# ipfs cat QmZ9Y8cmsyDnXyp3pYx3TtdBmQA1m4ku1p9dSVtBERRDiy > ciphertext.json

# # Visualizza
# cat ciphertext.json

# {"nonce":"abcd1234...","ct":"xyz789encrypted_data_here..."}
# Questo è ILLEGGIBILE - solo nonce e ciphertext base64

# Scarica il manifest
# ipfs cat QmXjHJDGRzsj8i3aRBvKhYNbUD1F2hUxarAZWHAST8TngL > manifest.json

# # Visualizza
# cat manifest.json | python -m json.tool

# {
#   "cipher_algo": "AES-256-GCM",
#   "ciphertext_cid": "QmZ9Y8cmsyDnXyp3pYx3TtdBmQA1m4ku1p9dSVtBERRDiy",
#   "group_signature": "long_base64_signature_here...",
#   "metadata": {
#     "node_id": "1",
#     "role": "participant",
#     "round": 3
#   },
#   "model_type": "FL_weights_encrypted",
#   "timestamp": 1729359991,
#   "version": "1.0"
# }
# Nota che il manifest NON contiene i pesi, solo metadati + CID del ciphertext.


# test_decrypt_without_key.py

# Il risultato dimostrerà che senza chiave Kc è impossibile leggere i pesi.
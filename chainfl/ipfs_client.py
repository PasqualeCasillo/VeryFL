"""
Simple IPFS client for VeryFL integration
Handles basic model storage and retrieval from IPFS network
"""
import ipfshttpclient
import pickle
import logging
import json
from collections import OrderedDict
import torch
import warnings
import base64
import time
import os
from typing import Optional, List, Dict, Tuple

# Import crypto utilities
from chainfl.crypto_bbs import (
    sha256, b64e, b64d,
    aesgcm_encrypt, aesgcm_decrypt,
    bbs_sign_messages, group_verify_any
)

# Disabilita warning versione IPFS
warnings.filterwarnings('ignore', message='Unsupported daemon version')

logger = logging.getLogger(__name__)

# chainfl/ipfs_client.py

class IPFSClient:
    def __init__(self, ipfs_api='/ip4/127.0.0.1/tcp/5001'):
        """
        Initialize IPFS client with cryptographic capabilities
        
        Args:
            ipfs_api: IPFS daemon API endpoint
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.client = ipfshttpclient.connect(ipfs_api)
            logger.debug("Connected to IPFS daemon")
        except Exception as e:
            logger.error(f"Failed to connect to IPFS: {e}")
            self.client = None
        
        # Group signature configuration
        self.group_public_keys = []
        self.encryption_enabled = False
    
    def set_group_keys(self, pubkeys: List[bytes]):
        """
        Configure group public keys for signature verification
        
        Args:
            pubkeys: List of BBS+ public keys for group members
        """
        self.group_public_keys = pubkeys
        logger.info(f"Configured {len(pubkeys)} group public keys")
    
    def enable_encryption(self, enabled: bool = True):
        """Enable/disable encryption for model uploads"""
        self.encryption_enabled = enabled
        logger.info(f"Encryption {'enabled' if enabled else 'disabled'}")
    
    def upload_model(self, model_state_dict, metadata=None):
        """
        LEGACY METHOD - Basic upload without encryption
        Maintained for backward compatibility
        """
        if self.client is None:
            logger.error("IPFS client not connected")
            return None
        
        try:
            model_data = {
                'state_dict': model_state_dict,
                'metadata': metadata or {}
            }
            
            model_bytes = pickle.dumps(model_data)
            result = self.client.add_bytes(model_bytes)
            ipfs_hash = result
            
            logger.info(f"Model uploaded to IPFS: {ipfs_hash}")
            return ipfs_hash
            
        except Exception as e:
            logger.error(f"Failed to upload model to IPFS: {e}")
            return None
    
    def upload_model_secured(
        self, 
        model_state_dict, 
        metadata: Optional[Dict] = None,
        node_secret_key: Optional[bytes] = None
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Upload con manifest STRUTTURATO + firma BBS+
        """
        if self.client is None:
            logger.error("IPFS client not connected")
            return None, None

        if self.encryption_enabled and node_secret_key is None:
            raise ValueError("node_secret_key required when encryption is enabled")

        try:
            # 1. Serialize model
            model_data = {
                'state_dict': model_state_dict,
                'metadata': metadata or {}
            }
            model_bytes = pickle.dumps(model_data)
            logger.debug(f"Serialized model: {len(model_bytes)} bytes")

            if self.encryption_enabled:
                # 2. Generate Kc
                Kc = os.urandom(32)
                logger.debug("Generated 256-bit encryption key")

                # 3. Encrypt
                encrypted_data = aesgcm_encrypt(Kc, model_bytes, aad=None)
                logger.debug(f"Encrypted model")

                # 4. Upload ciphertext
                ciphertext_json = json.dumps(encrypted_data, separators=(',', ':')).encode()
                # logger.info(f" ciphertext_json uploaded: {ciphertext_json}")
                cid_cipher = self.client.add_bytes(ciphertext_json)
                logger.info(f" Ciphertext uploaded: {cid_cipher}")

                # 5. Create STRUCTURED manifest (conforme al documento)
                manifest = {
                    'model_type': 'FL_weights_encrypted',
                    'version': '1.0',
                    'cipher_algo': 'AES-256-GCM',
                    'ciphertext_cid': cid_cipher,
                    'metadata': metadata or {},
                    'timestamp': int(time.time()),
                    'training_hash': sha256(model_bytes).hex()[:16]  # Checksum opzionale
                }

                # 6. Sign manifest with BBS+
                manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
                manifest_hash = sha256(manifest_bytes)

                signature = bbs_sign_messages([manifest_hash], node_secret_key)
                manifest['group_signature'] = b64e(signature)

                logger.info(f" Manifest signed with BBS+ group signature")

                # 7. Upload manifest
                manifest_final = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
                cid_manifest = self.client.add_bytes(manifest_final)

                logger.info(f" Signed manifest: {cid_manifest}")

                # 8. Return key data
                key_data = {
                    'cid_manifest': cid_manifest,
                    'cid_cipher': cid_cipher,
                    'Kc': b64e(Kc),
                    'encrypted': True,
                    'signature': b64e(signature)
                }

                return cid_manifest, key_data

            else:
                # Fallback non-encrypted
                ipfs_hash = self.client.add_bytes(model_bytes)
                key_data = {
                    'cid_manifest': ipfs_hash,
                    'encrypted': False
                }
                return ipfs_hash, key_data

        except Exception as e:
            logger.error(f"Upload failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None
    
    def download_model(self, ipfs_hash):
        """
        LEGACY METHOD - Basic download without verification
        Maintained for backward compatibility
        """
        if self.client is None:
            logger.error("IPFS client not connected")
            return None
        
        try:
            model_bytes = self.client.cat(ipfs_hash)
            model_data = pickle.loads(model_bytes)
            
            logger.info(f"Model downloaded from IPFS: {ipfs_hash}")
            return model_data['state_dict']
            
        except Exception as e:
            logger.error(f"Failed to download model from IPFS: {e}")
            return None
    
    def download_model_secured(
        self, 
        cid_manifest: str, 
        Kc: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Download and verify encrypted model
        
        Args:
            cid_manifest: IPFS CID of the signed manifest
            Kc: Base64url encoded encryption key (required if model is encrypted)
        
        Returns:
            model_state_dict or None if verification fails
        """
        if self.client is None:
            logger.error("IPFS client not connected")
            return None
        
        try:
            # Step 1: Download manifest
            manifest_bytes = self.client.cat(cid_manifest)
            manifest = json.loads(manifest_bytes)
            logger.debug(f"Downloaded manifest from {cid_manifest}")
            
            # Step 2: Verify group signature
            if 'group_signature' in manifest:
                signature = b64d(manifest.pop('group_signature'))
                manifest_to_verify = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
                manifest_hash = sha256(manifest_to_verify)
                
                if not group_verify_any(self.group_public_keys, [manifest_hash], signature):
                    raise ValueError("Group signature verification FAILED - not from authorized member")
                
                logger.info("Group signature verified - model from authorized member")
            else:
                logger.warning("No group signature in manifest - skipping verification")
            
            # Step 3: Check if encrypted
            if manifest.get('model_type') == 'FL_weights_encrypted':
                if Kc is None:
                    raise ValueError("Encryption key (Kc) required for encrypted model")
                
                # Step 4: Download ciphertext
                cid_cipher = manifest['ciphertext_cid']
                encrypted_data_json = self.client.cat(cid_cipher)
                encrypted_data = json.loads(encrypted_data_json)
                logger.debug(f"Downloaded ciphertext from {cid_cipher}")
                
                # Step 5: Decrypt
                Kc_bytes = b64d(Kc)
                model_bytes = aesgcm_decrypt(Kc_bytes, encrypted_data, aad=None)
                logger.info("Model decrypted successfully")
                
            else:
                # Non-encrypted model (legacy)
                model_bytes = self.client.cat(cid_manifest)
                logger.warning("Downloading non-encrypted model (legacy mode)")
            
            # Step 6: Deserialize
            model_data = pickle.loads(model_bytes)
            
            return model_data['state_dict']
            
        except ValueError as ve:
            logger.error(f"Verification failed: {ve}")
            return None
        except Exception as e:
            logger.error(f"Failed to download secured model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def pin_model(self, ipfs_hash):
        """
        Pin model to prevent garbage collection
        
        Args:
            ipfs_hash: IPFS hash to pin
        """
        if self.client is None:
            return False
        
        try:
            self.client.pin.add(ipfs_hash)
            logger.info(f"Model pinned: {ipfs_hash}")
            return True
        except Exception as e:
            logger.error(f"Failed to pin model: {e}")
            return False
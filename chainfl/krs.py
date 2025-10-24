# chainfl/krs.py
"""
Key Release Service (KRS) - Simplified PoC
Rilascia chiavi a membri verificati tramite BBS+ signature
"""
import os
import logging
from typing import Dict, Optional, List
from chainfl.crypto_bbs import sha256, group_verify_any, b64e, b64d

logger = logging.getLogger(__name__)

class KeyReleaseService:
    """
    KRS semplificato: verifica BBS+ e rilascia Kc in chiaro.
    
    Semplificazione PoC: 
    - NO HPKE wrapping (troppo complesso)
    - Kc ritornata direttamente dopo verifica firma
    - In produzione: usare HPKE per sicurezza
    """
    
    def __init__(self, group_public_keys: List[bytes]):
        self.group_public_keys = group_public_keys
        self.key_store: Dict[str, str] = {}  # {cid_manifest: Kc_base64}
        self.request_count = 0
        
        logger.info(f"🔑 KRS initialized with {len(group_public_keys)} group members")
    
    def register_key(self, cid_manifest: str, Kc_base64: str):
        """
        Registra chiave per un manifest.
        Chiamato dal publisher dopo upload.
        """
        self.key_store[cid_manifest] = Kc_base64
        logger.debug(f"KRS: Registered key for {cid_manifest[:10]}...")
    
    def request_key(
        self, 
        cid_manifest: str,
        nonce: str,
        group_signature: bytes
    ) -> Optional[str]:
        """
        Rilascia chiave a membro verificato.
        
        Returns:
            Kc (base64) if authorized, None otherwise
        """
        self.request_count += 1
        
        # 1. Verifica firma di gruppo
        message = sha256(f"{cid_manifest}||{nonce}".encode())
        
        is_valid = group_verify_any(
            self.group_public_keys, 
            [message], 
            group_signature
        )
        
        if not is_valid:
            logger.warning(f"❌ KRS: Invalid signature - request #{self.request_count} DENIED")
            return None
        
        logger.info(f"✓ KRS: Group member verified (request #{self.request_count})")
        
        # 2. Controlla se chiave esiste
        if cid_manifest not in self.key_store:
            logger.error(f"❌ KRS: No key for {cid_manifest[:10]}...")
            return None
        
        Kc = self.key_store[cid_manifest]
        
        logger.info(f"✓ KRS: Key released for {cid_manifest[:10]}... (request #{self.request_count})")
        
        return Kc
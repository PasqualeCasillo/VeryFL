# chainfl/krs.py
"""
Key Release Service (KRS) con HPKE encryption
Rilascia chiavi a membri verificati tramite BBS+ signature
"""
import os
import logging
from typing import Dict, Optional, List
from chainfl.crypto_bbs import (
    sha256, 
    group_verify_any, 
    b64e, 
    b64d,
    hpke_seal  # ← NUOVO IMPORT
)

logger = logging.getLogger(__name__)

class KeyReleaseService:
    """
    KRS con HPKE key wrapping per secure key release
    """
    
    def __init__(self, group_public_keys: List[bytes]):
        self.group_public_keys = group_public_keys
        self.key_store: Dict[str, str] = {}  # {cid_manifest: Kc_base64}
        self.request_count = 0
        
        # ← NUOVO: Rate limiting
        self.request_log: Dict[str, List[float]] = {}  # {requester_id: [timestamps]}
        
        logger.info(f" KRS initialized with {len(group_public_keys)} group members")
        logger.info(f" HPKE key wrapping ENABLED")
    
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
        group_signature: bytes,
        pk_eph_b64: str  # ← NUOVO: chiave pubblica effimera del richiedente
    ) -> Optional[Dict[str, str]]:
        """
        Rilascia chiave a membro verificato con HPKE wrapping.
        
        Args:
            cid_manifest: CID del manifest richiesto
            nonce: Nonce per freshness
            group_signature: Firma BBS+ su H(cid_manifest||nonce||pk_eph)
            pk_eph_b64: Chiave pubblica effimera X25519 del richiedente
        
        Returns:
            HPKE-wrapped key {pk_eph, nonce, ct} or None if unauthorized
        """
        self.request_count += 1
        
        # ← NUOVO: Rate limiting
        if not self._check_rate_limit(group_signature):
            logger.warning(f" KRS: Rate limit exceeded - request #{self.request_count} DENIED")
            return None
        
        # 1. Verifica firma di gruppo su messaggio completo
        message = sha256(f"{cid_manifest}||{nonce}||{pk_eph_b64}".encode())
        
        is_valid = group_verify_any(
            self.group_public_keys, 
            [message], 
            group_signature
        )
        
        if not is_valid:
            logger.warning(f" KRS: Invalid signature - request #{self.request_count} DENIED")
            return None
        
        logger.info(f" KRS: Group member verified (request #{self.request_count})")
        
        # 2. Controlla se chiave esiste
        if cid_manifest not in self.key_store:
            logger.error(f" KRS: No key for {cid_manifest[:10]}...")
            return None
        
        Kc_b64 = self.key_store[cid_manifest]
        
        # 3. ← NUOVO: Cifra chiave con HPKE verso pk_eph
        try:
            Kc_bytes = b64d(Kc_b64)  # Decodifica da base64
            
            wrapped = hpke_seal(
                pk_recipient_b64=pk_eph_b64,
                plaintext=Kc_bytes,
                info=b"veryfl-krs-v1"  # Context string
            )
            
            logger.info(f" KRS: Key wrapped with HPKE for {cid_manifest[:10]}... (request #{self.request_count})")
            logger.debug(f"   Ephemeral key: {pk_eph_b64[:20]}...")
            
            return wrapped  # {pk_eph, nonce, ct}
            
        except Exception as e:
            logger.error(f" KRS: HPKE wrapping failed: {e}")
            return None
    
    def _check_rate_limit(self, group_signature: bytes, max_requests: int = 10, window_seconds: int = 3600) -> bool:
        """
        Rate limiter: max 10 richieste/ora per firma
        
        Args:
            group_signature: Firma del richiedente (usata come pseudo-ID)
            max_requests: Max richieste consentite
            window_seconds: Finestra temporale in secondi
        
        Returns:
            True se sotto il limite, False altrimenti
        """
        import time
        
        # Crea pseudo-ID anonimo dalla firma
        requester_id = sha256(group_signature).hex()[:16]
        
        now = time.time()
        
        # Pulisci vecchie entry
        if requester_id in self.request_log:
            recent = [t for t in self.request_log[requester_id] if now - t < window_seconds]
            self.request_log[requester_id] = recent
        else:
            self.request_log[requester_id] = []
        
        # Controlla limite
        if len(self.request_log[requester_id]) >= max_requests:
            logger.warning(f"  Rate limit hit for requester {requester_id} ({len(self.request_log[requester_id])} requests in {window_seconds}s)")
            return False
        
        # Registra richiesta
        self.request_log[requester_id].append(now)
        return True
# chainfl/revocation.py
"""
Verifier-Local Revocation (VLR) - Minimal PoC
"""
import json
import logging
from pathlib import Path
from typing import Set, List

logger = logging.getLogger(__name__)

class VLRManager:
    """Gestisce lista nodi revocati (file JSON locale)"""
    
    def __init__(self, vlr_file='revocation_list.json'):
        self.vlr_file = Path(vlr_file)
        self.revoked_nodes: Set[str] = set()
        self._load()
    
    def _load(self):
        """Carica revocation list da file"""
        if self.vlr_file.exists():
            try:
                with open(self.vlr_file, 'r') as f:
                    data = json.load(f)
                    self.revoked_nodes = set(data.get('revoked', []))
                    logger.info(f"VLR loaded: {len(self.revoked_nodes)} revoked nodes")
            except Exception as e:
                logger.warning(f"Failed to load VLR: {e}")
    
    def _save(self):
        """Salva revocation list su file"""
        try:
            with open(self.vlr_file, 'w') as f:
                json.dump({'revoked': list(self.revoked_nodes)}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save VLR: {e}")
    
    def revoke(self, node_id: str):
        """Revoca un nodo"""
        self.revoked_nodes.add(node_id)
        self._save()
        logger.warning(f"  NODE REVOKED: {node_id}")
    
    def is_revoked(self, node_id: str) -> bool:
        """Verifica se nodo è revocato"""
        return node_id in self.revoked_nodes
    
    def get_revoked(self) -> List[str]:
        """Ottieni lista nodi revocati"""
        return list(self.revoked_nodes)
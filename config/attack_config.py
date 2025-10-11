# config/attack_config.py
import logging

logger = logging.getLogger(__name__)

class AttackConfig:
    """Configurazione centralizzata per simulare attacchi Byzantine"""
    
    def __init__(self, attack_type='none', byzantine_ratio=0.0, attack_start_round=0):
        """
        Args:
            attack_type: 'none', 'label_flipping', 'model_poisoning'
            byzantine_ratio: frazione di nodi malevoli (0.0 - 1.0)
            attack_start_round: round da cui iniziare l'attacco
        """
        self.attack_type = attack_type
        self.byzantine_ratio = byzantine_ratio
        self.attack_start_round = attack_start_round
        
        logger.info(f"Attack Config: type={attack_type}, ratio={byzantine_ratio}, start={attack_start_round}")
    
    def is_active(self, current_round):
        """Verifica se l'attacco è attivo in questo round"""
        return self.attack_type != 'none' and current_round >= self.attack_start_round
    
    def should_attack(self, node_id, total_nodes):
        """Determina se questo nodo è Byzantine"""
        if self.byzantine_ratio == 0:
            return False
        
        byzantine_count = int(total_nodes * self.byzantine_ratio)
        
        # FIX: node_id è stringa "1", "2", etc.
        # Converti a int e sottrai 1 per confronto (node_id parte da 1, non 0)
        node_index = int(node_id) - 1
        
        return node_index < byzantine_count
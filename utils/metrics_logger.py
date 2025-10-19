# utils/metrics_logger.py

import json
from pathlib import Path

class MetricsLogger:
    def __init__(self, save_dir='results'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.metrics = {
            'rounds': [],
            'loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': [],
            'per_node': {},
            
            # NUOVO: Metriche attacco
            'byzantine_nodes': [],  # Lista node_id Byzantine per round
            'attack_active': [],    # Boolean se attacco attivo
            'honest_avg_loss': [],  # Loss media nodi onesti
            'byzantine_avg_loss': [], # Loss media nodi Byzantine
        }
    
    def log_round(self, round_num, global_metrics, node_metrics, loss, 
                  byzantine_nodes=None, attack_active=False, pre_aggregation_metrics=None):
        """
        Aggiungi parametri per tracking attacco
        
        Args:
            byzantine_nodes: lista node_id Byzantine
            attack_active: se attacco è attivo questo round
        """
        self.metrics['rounds'].append(round_num)
        self.metrics['loss'].append(float(loss))
        self.metrics['accuracy'].append(global_metrics['accuracy'])
        self.metrics['precision'].append(global_metrics['precision'])
        self.metrics['recall'].append(global_metrics['recall'])
        self.metrics['f1'].append(global_metrics['f1'])
        self.metrics['auc'].append(global_metrics['auc'])
        
        # NUOVO: Log attacco
        self.metrics['attack_active'].append(attack_active)
        self.metrics['byzantine_nodes'].append(byzantine_nodes if byzantine_nodes else [])
        
        # Calcola loss medie separate
        honest_losses = []
        byzantine_losses = []
        
        for node_id, metrics in node_metrics.items():
            if 'loss' in metrics:
                if byzantine_nodes and node_id in byzantine_nodes:
                    byzantine_losses.append(metrics['loss'])
                else:
                    honest_losses.append(metrics['loss'])
        
        self.metrics['honest_avg_loss'].append(
            sum(honest_losses) / len(honest_losses) if honest_losses else 0.0
        )
        self.metrics['byzantine_avg_loss'].append(
            sum(byzantine_losses) / len(byzantine_losses) if byzantine_losses else 0.0
        )
        
        # Per-node metrics
        for node_id, metrics in node_metrics.items():
            if node_id not in self.metrics['per_node']:
                self.metrics['per_node'][node_id] = {
                    k: [] for k in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'loss']
                }
            for k, v in metrics.items():
                if k in self.metrics['per_node'][node_id]:
                    self.metrics['per_node'][node_id][k].append(v)
    
    def save(self, filename):
        with open(self.save_dir / filename, 'w') as f:
            json.dump(self.metrics, f, indent=2)
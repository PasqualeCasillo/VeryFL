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
            'per_node': {}
        }
    
    def log_round(self, round_num, global_metrics, node_metrics, loss):
        self.metrics['rounds'].append(round_num)
        self.metrics['loss'].append(float(loss))
        self.metrics['accuracy'].append(global_metrics['accuracy'])
        self.metrics['precision'].append(global_metrics['precision'])
        self.metrics['recall'].append(global_metrics['recall'])
        self.metrics['f1'].append(global_metrics['f1'])
        self.metrics['auc'].append(global_metrics['auc'])
        
        for node_id, metrics in node_metrics.items():
            if node_id not in self.metrics['per_node']:
                self.metrics['per_node'][node_id] = {k: [] for k in ['accuracy', 'precision', 'recall', 'f1', 'auc']}
            for k, v in metrics.items():
                self.metrics['per_node'][node_id][k].append(v)
    
    def save(self, filename):
        with open(self.save_dir / filename, 'w') as f:
            json.dump(self.metrics, f, indent=2)
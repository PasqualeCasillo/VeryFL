# node/DecentralizedNode.py
import logging
from typing import Dict, Any
from copy import deepcopy
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from client.base.baseTrainer import BaseTrainer

logger = logging.getLogger(__name__)

class DecentralizedNode:
    def __init__(self, node_id: str, model: nn.Module, dataloader: DataLoader, 
                 trainer_class: BaseTrainer, train_args: dict, test_dataloader: DataLoader = None):
        self.node_id = node_id
        self.model = deepcopy(model)
        self.dataloader = dataloader
        self.trainer_class = trainer_class
        self.train_args = train_args
        self.test_dataloader = test_dataloader
        self.role = "participant"
        self.current_round = 0
        
        # Verifica distribuzione corretta
        labels = []
        for _, label in self.dataloader:
            labels.extend(label.tolist())
        
        from collections import Counter
        class_dist = Counter(labels)
        logger.info(f"Node {node_id} class distribution: {class_dist}")
        logger.info(f"Node {node_id} dataset size: {len(labels)}")
        
        # Node capabilities
        self.compute_power = self._calculate_compute_power()
        self.bandwidth = self._calculate_bandwidth()
        self.reliability = self._calculate_reliability()
        self.data_size = len(labels)
        
    def _calculate_compute_power(self) -> int:
        """Calculate node's computational capacity"""
        total_params = sum(p.numel() for p in self.model.parameters())
        return min(total_params // 1000, 10000)
        
    def _calculate_bandwidth(self) -> int:
        """Calculate node's bandwidth capacity"""
        return 1000  # Mbps
        
    def _calculate_reliability(self) -> int:
        """Calculate node's reliability score"""
        return 95  # 95% uptime
        
    def set_role(self, role: str, round_num: int):
        """Set node role for current round"""
        self.role = role
        self.current_round = round_num
        logger.info(f"Node {self.node_id} assigned role: {role} for round {round_num}")
        
    def train_local_model(self) -> Dict[str, Any]:
        """Train local model and return results with metrics"""
        logger.info(f"Node {self.node_id} starting local training for round {self.current_round}")
        
        # Metriche pre-training
        initial_loss, initial_acc = self._evaluate_model()
        
        trainer = self.trainer_class(
            model=self.model,
            dataloader=self.dataloader,
            criterion=torch.nn.CrossEntropyLoss(),
            args=self.train_args
        )
        
        results = trainer.train(self.train_args.get('num_steps', 1))
        
        # Metriche post-training
        final_loss, final_acc = self._evaluate_model()
        
        logger.info(f"Node {self.node_id} training results: "
                   f"initial_loss={initial_loss:.4f}, "
                   f"final_loss={final_loss:.4f}, "
                   f"accuracy={final_acc:.2f}%")
        
        return {
            'round': self.current_round,
            'node_id': self.node_id,
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'initial_accuracy': initial_acc,
            'final_accuracy': final_acc,
            'results': results
        }
        
    def _evaluate_model(self) -> tuple:
        """Calcola loss e accuracy"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.dataloader:
                data, target = data.to(self.train_args['device']), target.to(self.train_args['device'])
                output = self.model(data)
                total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
        
    def get_model_state_dict(self):
        """Get current model state dict"""
        return self.model.state_dict()
        
    def load_state_dict(self, state_dict):
        """Load new model state dict"""
        self.model.load_state_dict(state_dict)
        
    def test_model(self) -> Dict[str, Any]:
        """Test current model if test data available"""
        if self.test_dataloader is None:
            return {}
            
        self.model.eval()
        total_loss = 0
        correct = 0
        num_data = 0
        
        with torch.no_grad():
            for batch_id, (data, targets) in enumerate(self.test_dataloader):
                data, targets = data.to(self.train_args['device']), targets.to(self.train_args['device'])
                output = self.model(data)
                total_loss += torch.nn.functional.cross_entropy(output, targets, reduction='sum').item()
                pred = output.data.max(1)[1]
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
                num_data += output.size(0)
                
        acc = 100.0 * (float(correct) / float(num_data))
        total_l = total_loss / float(num_data)
        
        return {
            'node_id': self.node_id,
            'round': self.current_round,
            'loss': total_l,
            'acc': acc
        }
        
    def get_auction_offer(self, base_cost: int = 100) -> Dict[str, int]:
        # Varia le offerte tra nodi e round
        noise_factor = random.uniform(0.8, 1.2)
        return {
            'computePower': int(self.compute_power * noise_factor),
            'bandwidth': int(self.bandwidth * noise_factor),
            'reliability': self.reliability,
            'dataSize': self.data_size,
            'cost': int(base_cost * random.uniform(0.7, 1.3))  # Varia costo
        }
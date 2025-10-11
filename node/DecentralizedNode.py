# node/DecentralizedNode.py
import logging
from typing import Dict, Any, List 
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
        #logger.info(f"Node {node_id} class distribution: {class_dist}")
        #logger.info(f"Node {node_id} dataset size: {len(labels)}")
        logger.debug(f"Node {node_id}: {len(labels)} samples, dist={dict(class_dist)}")
        
        # Node capabilities
        self.compute_power = self._calculate_compute_power()
        self.bandwidth = self._calculate_bandwidth()
        self.reliability = self._calculate_reliability()
        self.data_size = len(labels)
        
    async def upload_model_to_blockchain_and_ipfs(self, ipfs_client, blockchain_proxy, auction_address):
        """
        Upload model: Node → IPFS → Blockchain
        Returns True if successful, False otherwise
        """
        try:
            logger.info(f"Node {self.node_id} uploading model to IPFS...")
            
            model_data = self.get_model_state_dict()
            ipfs_hash = ipfs_client.upload_model(model_data, metadata={
                'node_id': self.node_id,
                'round': self.current_round
            })
            
            if not ipfs_hash:
                logger.error(f"Node {self.node_id} failed IPFS upload")
                return False
            
            logger.info(f"Node {self.node_id} uploaded to IPFS: {ipfs_hash}")
            
            # Register hash on blockchain
            import brownie
            node_index = int(self.node_id)
            node_account = brownie.accounts[node_index]
            
            contracts = brownie.project.chainServer
            auction_contract = contracts.AggregatorAuction.at(auction_address)
            
            tx = auction_contract.submitModelHash(ipfs_hash, {'from': node_account})
            tx.wait(1)  # Wait for 1 confirmation
            
            logger.info(f"Node {self.node_id} registered hash on blockchain (block {tx.block_number})")
            
            return True
            
        except Exception as e:
            logger.error(f"Node {self.node_id} upload failed: {e}")
            return False
        
    async def aggregate_from_ipfs(self, ipfs_client, blockchain_proxy, auction_address):
        """
        L'aggregatore scarica i modelli da IPFS usando gli hash dalla blockchain.
        NESSUN intermediario: Blockchain → IPFS → Aggregatore
        """
        logger.info(f"Node {self.node_id} (AGGREGATOR) starting aggregation from IPFS")
        
        try:
            # 1. Leggi gli hash dalla blockchain
            import brownie
            contracts = brownie.project.chainServer
            auction_contract = contracts.AggregatorAuction.at(auction_address)
            
            ipfs_hashes = auction_contract.getAllModelHashes()
            logger.info(f"Aggregator found {len(ipfs_hashes)} model hashes on blockchain")
            
            # 2. Scarica i modelli da IPFS
            downloaded_models = []
            
            for idx, ipfs_hash in enumerate(ipfs_hashes):
                if not ipfs_hash:  # Skip vuoti
                    continue
                
                # Filtra il proprio modello (opzionale)
                node_index = int(self.node_id)
                if idx == node_index:
                    # Usa il proprio modello già in memoria
                    downloaded_models.append(self.get_model_state_dict())
                    logger.info(f"Using own model for aggregation")
                else:
                    # Scarica dagli altri nodi
                    logger.info(f"Downloading model from IPFS: {ipfs_hash[:10]}...")
                    model_data = ipfs_client.download_model(ipfs_hash)
                    
                    if model_data:
                        downloaded_models.append(model_data)
                        logger.info(f"✓ Downloaded model {idx+1}")
                    else:
                        logger.warning(f"✗ Failed to download model {idx+1}")
            
            logger.info(f"Aggregator downloaded {len(downloaded_models)} models")
            
            # 3. Aggrega (FedAvg)
            aggregated_state = {}
            num_models = len(downloaded_models)
            
            for key in downloaded_models[0].keys():
                aggregated_state[key] = sum(
                    model[key] for model in downloaded_models
                ) / num_models
            
            logger.info(f" Node {self.node_id} completed aggregation from IPFS")
            
            return aggregated_state
            
        except Exception as e:
            logger.error(f"Aggregator IPFS aggregation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None    
    
    async def upload_global_model_to_ipfs(self, ipfs_client, aggregated_model, auction_address):
        """
        L'AGGREGATORE uploada il modello globale su IPFS e registra l'hash sulla blockchain.
        Chiamato SOLO dal nodo aggregatore dopo l'aggregazione.
        """
        try:
            logger.info(f"Aggregator {self.node_id} uploading GLOBAL model to IPFS...")

            # 1. Upload su IPFS
            ipfs_hash = ipfs_client.upload_model(aggregated_model, metadata={
                'type': 'global_aggregated',
                'round': self.current_round,
                'aggregator': self.node_id
            })

            if not ipfs_hash:
                logger.error(f" Aggregator failed IPFS upload")
                return None

            logger.info(f" Global model uploaded to IPFS: {ipfs_hash}")

            # 2. Registra hash sulla blockchain
            import brownie
            node_index = int(self.node_id)
            aggregator_account = brownie.accounts[node_index]

            contracts = brownie.project.chainServer
            auction_contract = contracts.AggregatorAuction.at(auction_address)

            # CHIAMA submitGlobalModel() del contratto
            tx = auction_contract.submitGlobalModel(ipfs_hash, {'from': aggregator_account})
            logger.info(f"Global hash registered on blockchain: {tx.txid}")

            return ipfs_hash

        except Exception as e:
            logger.error(f"Aggregator upload failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        
    async def download_global_model_from_ipfs(self, ipfs_client, auction_address):
        """
        Ogni nodo AUTONOMAMENTE scarica il modello globale da IPFS.
        Legge l'hash dalla blockchain e scarica da IPFS.
        """
        try:
            logger.info(f"Node {self.node_id} downloading GLOBAL model from IPFS...")

            # 1. Leggi hash dalla blockchain
            import brownie
            contracts = brownie.project.chainServer
            auction_contract = contracts.AggregatorAuction.at(auction_address)

            # CHIAMA getGlobalModelHash() del contratto
            global_ipfs_hash = auction_contract.getGlobalModelHash()

            if not global_ipfs_hash:
                logger.error(f"No global model hash on blockchain")
                return False

            logger.info(f"Read global hash from blockchain: {global_ipfs_hash[:10]}...")

            # 2. Scarica da IPFS
            global_model_data = ipfs_client.download_model(global_ipfs_hash)

            if not global_model_data:
                logger.error(f" Failed to download from IPFS")
                return False

            # 3. Carica nel proprio modello locale
            self.load_state_dict(global_model_data)
            logger.info(f"Node {self.node_id} loaded global model successfully")

            return True

        except Exception as e:
            logger.error(f" Node {self.node_id} download failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
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
        
        self._last_training_loss = final_loss
        
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
        noise_factor = random.uniform(0.8, 1.2)
        return {
            'computePower': int(self.compute_power * noise_factor),
            'bandwidth': int(self.bandwidth * noise_factor),
            'reliability': self.reliability,
            'dataSize': self.data_size,
            'cost': max(1, int(base_cost * random.uniform(0.7, 1.3)))  # Garantisce cost >= 1
        }
        
    def aggregate_models(self, other_nodes_models: List[Dict]) -> Dict:
        """
        Perform FedAvg aggregation on received models.
        This method is called ONLY if this node is the elected aggregator.

        Args:
            other_nodes_models: List of state_dicts from other nodes

        Returns:
            Aggregated state_dict (global model)
        """
        logger.info(f"Node {self.node_id} executing aggregation (elected aggregator)")

        logger.warning(f"NODE {self.node_id} IS PERFORMING AGGREGATION")
        logger.warning(f"Role: {self.role}")
        logger.warning(f"Models received: {len(other_nodes_models)}")
        # Include own model in aggregation
        all_models = other_nodes_models + [self.get_model_state_dict()]
        num_models = len(all_models)

        logger.info(f"Aggregating {num_models} models (including own)")

        # FedAvg: simple averaging
        aggregated_state = {}
        for key in all_models[0].keys():
            # Sum all model parameters
            aggregated_state[key] = sum(model[key] for model in all_models) / num_models

        logger.info(f"Aggregation complete - {len(aggregated_state)} parameters updated")
        return aggregated_state
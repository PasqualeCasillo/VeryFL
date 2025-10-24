# node/DecentralizedNode.py
import logging
from typing import Dict, Any, List 
from copy import deepcopy
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from client.base.baseTrainer import BaseTrainer
from utils.attack_utils import create_flipped_dataloader
from chainfl.ipfs_client import IPFSClient
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)

class DecentralizedNode:
    def __init__(self, node_id: str, model: nn.Module, dataloader: DataLoader, 
                 trainer_class: BaseTrainer, train_args: dict, test_dataloader: DataLoader = None,
                 num_classes: int = 10):
        self.node_id = node_id
        self.model = deepcopy(model)
        self.dataloader = dataloader
        self.trainer_class = trainer_class
        self.train_args = train_args
        self.test_dataloader = test_dataloader
        self.role = "participant"
        self.current_round = 0
        self.is_byzantine = False
        self.attack_config = None
        self.num_classes = num_classes
        
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
        Upload model: Node → IPFS (encrypted) → Blockchain
        VERSIONE CON ENCRYPTION E GROUP SIGNATURE
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Node {self.node_id} uploading encrypted model to IPFS...")
            
            model_data = self.get_model_state_dict()
            metadata = {
                'node_id': self.node_id,
                'round': self.current_round,
                'role': self.role
            }
            
            # Upload con encryption e firma gruppo
            cid_manifest, key_data = ipfs_client.upload_model_secured(
                model_data,
                metadata,
                self.bbs_secret_key
            )
            
            if not cid_manifest:
                logger.error(f"Node {self.node_id} failed IPFS upload")
                return False
            
            logger.info(f"Node {self.node_id} uploaded to IPFS: {cid_manifest}")
            
            # Store encryption key locally
            if key_data and 'Kc' in key_data:
                self.store_encryption_key(cid_manifest, key_data['Kc'])
            
            # Register hash on blockchain
            import brownie
            node_index = int(self.node_id)
            node_account = brownie.accounts[node_index]
            
            contracts = brownie.project.chainServer
            auction_contract = contracts.AggregatorAuction.at(auction_address)
            
            tx = auction_contract.submitModelHash(cid_manifest, {'from': node_account})
            tx.wait(1)
            
            logger.info(f"Node {self.node_id} registered hash on blockchain (block {tx.block_number})")
            
            return True
            
        except Exception as e:
            logger.error(f"Node {self.node_id} upload failed: {e}")
            return False
        
    async def aggregate_from_ipfs(self, ipfs_client, blockchain_proxy, auction_address, aggregation_method='fedavg'):
        """
        L'aggregatore scarica i modelli da IPFS usando gli hash dalla blockchain
        e li aggrega usando il metodo specificato.
        VERSIONE CON VERIFICA CRITTOGRAFICA
        
        Args:
            ipfs_client: Client IPFS con encryption abilitata
            blockchain_proxy: Proxy blockchain
            auction_address: Indirizzo contratto auction
            aggregation_method: 'fedavg', 'krum', or 'median'
        
        Returns:
            Aggregated state_dict
        """
        logger.info(f"Node {self.node_id} (AGGREGATOR) starting aggregation from IPFS")
        logger.info(f"Aggregation method: {aggregation_method.upper()}")
        
        try:
            # 1. Leggi gli hash dalla blockchain
            import brownie
            contracts = brownie.project.chainServer
            auction_contract = contracts.AggregatorAuction.at(auction_address)
            
            ipfs_hashes = auction_contract.getAllModelHashes()
            logger.info(f"Aggregator found {len(ipfs_hashes)} model hashes on blockchain")
            
            # 2. Scarica i modelli da IPFS con verifica crittografica
            downloaded_models = []
            
            for idx, ipfs_hash in enumerate(ipfs_hashes):
                if not ipfs_hash:
                    continue
                
                # Filtra il proprio modello (opzionale)
                node_index = int(self.node_id)
                if idx == node_index:
                    # Usa il proprio modello già in memoria
                    downloaded_models.append(self.get_model_state_dict())
                    logger.info(f"Using own model for aggregation")
                else:
                    # Scarica dagli altri nodi CON VERIFICA
                    logger.info(f"Downloading encrypted model from IPFS: {ipfs_hash[:10]}...")
                    
                    # NUOVO: Recupera chiave di cifratura dal nodo sorgente
                    # In un sistema reale, questo richiederebbe un KRS o key distribution protocol
                    # Per il PoC, assumiamo che l'aggregatore abbia accesso alle chiavi
                    
                    # Opzione 1: Chiave condivisa tra nodi (semplificazione PoC)
                    # Opzione 2: Key Release Service
                    # Opzione 3: Chiavi in smart contract cifrate
                    
                    # Per ora, proviamo a scaricare e verificare la firma
                    # Se il modello è cifrato, serve la chiave Kc
                    
                    try:
                        # Tentativo 1: Scarica come modello sicuro con verifica firma
                        # Nota: serve Kc per decifrare. Per il PoC, lo recuperiamo da una fonte trusted
                        
                        # WORKAROUND PoC: Cerca chiave in memoria del nodo locale
                        # In produzione, useremmo un KRS con challenge-response
                        Kc = self._get_encryption_key_for_manifest(ipfs_hash)
                        
                        if Kc:
                            model_data = ipfs_client.download_model_secured(ipfs_hash, Kc)
                        else:
                            # Fallback: prova download senza encryption (legacy)
                            logger.warning(f"No encryption key for {ipfs_hash}, trying legacy download")
                            model_data = ipfs_client.download_model(ipfs_hash)
                        
                        if model_data:
                            downloaded_models.append(model_data)
                            logger.info(f"Downloaded and verified model {idx+1}")
                        else:
                            logger.warning(f"Failed to download model {idx+1}")
                            
                    except Exception as e:
                        logger.error(f"Error downloading model {idx+1}: {e}")
                        continue
            
            logger.info(f"Aggregator downloaded {len(downloaded_models)} models")
            
            # 3. Aggrega usando metodo configurabile
            if aggregation_method.lower() == 'krum':
                logger.info("=" * 60)
                logger.info("USING KRUM AGGREGATION")
                logger.info("=" * 60)
                
                from server.aggregation_alg.krum import krumAggregator
                
                aggregator = krumAggregator(byzantine_ratio=0.3)
                aggregated_state = aggregator._aggregate_alg(downloaded_models)
                
                logger.info("Krum aggregation complete")
                
            elif aggregation_method.lower() == 'median':
                logger.info("Using MEDIAN aggregation")
                
                from server.aggregation_alg.median import medianAggregator
                aggregator = medianAggregator()
                aggregated_state = aggregator._aggregate_alg(downloaded_models)
                
            else:
                logger.info("Using FEDAVG aggregation")
                
                aggregated_state = {}
                num_models = len(downloaded_models)
                
                for key in downloaded_models[0].keys():
                    aggregated_state[key] = sum(
                        model[key] for model in downloaded_models
                    ) / num_models
            
            logger.info(f"Node {self.node_id} completed aggregation from IPFS")
            
            return aggregated_state
            
        except Exception as e:
            logger.error(f"Aggregator IPFS aggregation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None  
    
    async def upload_global_model_to_ipfs(self, ipfs_client, aggregated_model, auction_address):
        """
        L'AGGREGATORE uploada il modello globale su IPFS e registra l'hash sulla blockchain.
        VERSIONE CON ENCRYPTION E GROUP SIGNATURE
        """
        try:
            logger.info(f"Aggregator {self.node_id} uploading GLOBAL model to IPFS...")

            metadata = {
                'type': 'global_aggregated',
                'round': self.current_round,
                'aggregator': self.node_id
            }

            # Upload con encryption e firma gruppo
            cid_manifest, key_data = ipfs_client.upload_model_secured(
                aggregated_model,
                metadata,
                self.bbs_secret_key
            )

            if not cid_manifest:
                logger.error(f"Aggregator failed IPFS upload")
                return None

            logger.info(f"Global model uploaded to IPFS: {cid_manifest}")

            # Store encryption key
            if key_data and 'Kc' in key_data:
                self.store_encryption_key(cid_manifest, key_data['Kc'])

            # Registra hash sulla blockchain
            import brownie
            node_index = int(self.node_id)
            aggregator_account = brownie.accounts[node_index]

            contracts = brownie.project.chainServer
            auction_contract = contracts.AggregatorAuction.at(auction_address)

            tx = auction_contract.submitGlobalModel(cid_manifest, {'from': aggregator_account})
            logger.info(f"Global hash registered on blockchain: {tx.txid}")

            return cid_manifest

        except Exception as e:
            logger.error(f"Aggregator upload failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        
    async def download_global_model_from_ipfs(self, ipfs_client, auction_address):
        """
        Ogni nodo AUTONOMAMENTE scarica il modello globale da IPFS.
        Legge l'hash dalla blockchain e scarica da IPFS.
        VERSIONE CON DECRYPTION E VERIFICA
        """
        try:
            logger.info(f"Node {self.node_id} downloading GLOBAL model from IPFS...")

            # 1. Leggi hash dalla blockchain
            import brownie
            contracts = brownie.project.chainServer
            auction_contract = contracts.AggregatorAuction.at(auction_address)

            global_ipfs_hash = auction_contract.getGlobalModelHash()

            if not global_ipfs_hash:
                logger.error(f"No global model hash on blockchain")
                return False

            logger.info(f"Read global hash from blockchain: {global_ipfs_hash[:10]}...")

            # 2. Recupera chiave di cifratura
            Kc = self._get_encryption_key_for_manifest(global_ipfs_hash)

            # 3. Scarica da IPFS con verifica
            if Kc:
                global_model_data = ipfs_client.download_model_secured(global_ipfs_hash, Kc)
            else:
                # Fallback: prova senza encryption
                logger.warning("No encryption key, trying legacy download")
                global_model_data = ipfs_client.download_model(global_ipfs_hash)

            if not global_model_data:
                logger.error(f"Failed to download from IPFS")
                return False

            # 4. Carica nel proprio modello locale
            self.load_state_dict(global_model_data)
            logger.info(f"Node {self.node_id} loaded global model successfully")

            return True

        except Exception as e:
            logger.error(f"Node {self.node_id} download failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        
    def _get_encryption_key_for_manifest(self, cid_manifest: str) -> Optional[str]:
        """
        Recupera la chiave di cifratura per un manifest CID
        
        NOTA PoC: In un sistema reale, questo userebbe un Key Release Service (KRS)
        con proof of membership anonima. Per il PoC, usiamo storage locale.
        
        Args:
            cid_manifest: CID del manifest da cui recuperare la chiave
        
        Returns:
            Chiave Kc (base64url) o None se non disponibile
        """
        # Check in-memory storage
        if hasattr(self, '_encryption_keys') and cid_manifest in self._encryption_keys:
            return self._encryption_keys[cid_manifest]
        
        # WORKAROUND PoC: In assenza di KRS, l'aggregatore ha accesso a tutte le chiavi
        # In produzione, implementare challenge-response con KRS
        logger.debug(f"No local key found for {cid_manifest}")
        return None
    
    def store_encryption_key(self, cid_manifest: str, Kc: str):
        """
        Store encryption key for a manifest
        
        Args:
            cid_manifest: Manifest CID
            Kc: Encryption key (base64url encoded)
        """
        if not hasattr(self, '_encryption_keys'):
            self._encryption_keys = {}
        
        self._encryption_keys[cid_manifest] = Kc
        logger.debug(f"Stored encryption key for {cid_manifest}")
        
    def share_encryption_key_with(self, other_node: 'DecentralizedNode', cid_manifest: str):
        """
        Share encryption key with another node (PoC simplification)
        
        In produzione, questo sarebbe gestito da un KRS con:
        - Challenge-response protocol
        - Proof of group membership
        - Key wrapping con HPKE
        
        Args:
            other_node: Node to share key with
            cid_manifest: Manifest CID whose key to share
        """
        if hasattr(self, '_encryption_keys') and cid_manifest in self._encryption_keys:
            Kc = self._encryption_keys[cid_manifest]
            other_node.store_encryption_key(cid_manifest, Kc)
            logger.debug(f"Shared key for {cid_manifest} with Node {other_node.node_id}")
    
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
        """Train local model (con possibile poisoning)"""
        logger.info(f"Node {self.node_id} starting local training for round {self.current_round}")
        
        use_poisoned_data = (
            self.is_byzantine and 
            self.attack_config and 
            self.attack_config.is_active(self.current_round) and
            self.attack_config.attack_type == 'label_flipping'
        )
        
        if use_poisoned_data:
            logger.warning(f"Node {self.node_id} POISONING data with label flipping")
            from utils.attack_utils import create_flipped_dataloader
            
            training_dataloader = create_flipped_dataloader(
                self.dataloader,
                flip_probability=1.0,
                num_classes=self.num_classes  # FIX: Usa num_classes del nodo
            )
        else:
            training_dataloader = self.dataloader
        
        # Metriche pre-training
        initial_loss, initial_acc = self._evaluate_model()
        
        # Training
        trainer = self.trainer_class(
            model=self.model,
            dataloader=training_dataloader,
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
            'is_byzantine': self.is_byzantine,
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
        
    def configure_attack(self, attack_config, total_nodes):
        """
        Configura comportamento Byzantine per questo nodo
        
        Args:
            attack_config: AttackConfig instance
            total_nodes: numero totale di nodi
        """
        self.attack_config = attack_config
        self.is_byzantine = attack_config.should_attack(self.node_id, total_nodes)
        
        if self.is_byzantine:
            logger.warning(f"Node {self.node_id} configured as BYZANTINE")
        else:
            logger.debug(f"Node {self.node_id} configured as HONEST")
        
    def aggregate_models(self, other_nodes_models: List[Dict], method='fedavg') -> Dict:
        """
        Perform aggregation with configurable method.
        
        Args:
            other_nodes_models: List of state_dicts from other nodes
            method: 'fedavg', 'krum', or 'median'
            
        Returns:
            Aggregated state_dict (global model)
        """
        logger.info(f"Node {self.node_id} executing {method.upper()} aggregation")
        
        # Include own model
        all_models = other_nodes_models + [self.get_model_state_dict()]
        num_models = len(all_models)
        
        logger.info(f"Aggregating {num_models} models (including own)")
        
        # Scegli metodo di aggregazione
        if method.lower() == 'krum':
            from server.aggregation_alg.krum import krumAggregator
            
            # Calcola byzantine_ratio dai modelli disponibili
            # Assumi che ~25% siano Byzantine (configurable)
            aggregator = krumAggregator(byzantine_ratio=0.25)
            aggregated_state = aggregator._aggregate_alg(all_models)
            
            logger.info(" Krum aggregation complete")
            
        elif method.lower() == 'median':
            from server.aggregation_alg.median import medianAggregator
            
            aggregator = medianAggregator()
            aggregated_state = aggregator._aggregate_alg(all_models)
            
            logger.info(" Median aggregation complete")
            
        else:  # Default: FedAvg
            aggregated_state = {}
            
            for key in all_models[0].keys():
                # Simple averaging
                aggregated_state[key] = sum(model[key] for model in all_models) / num_models
            
            logger.info(" FedAvg aggregation complete")
        
        logger.info(f"Aggregated {len(aggregated_state)} parameters")
        return aggregated_state
    
    def request_key_from_krs(
        self, 
        krs,  # KeyReleaseService instance
        cid_manifest: str
    ) -> Optional[str]:
        """
        Richiedi chiave da KRS usando BBS+ signature.

        Returns:
            Kc (base64) or None
        """
        import os
        from chainfl.crypto_bbs import sha256, bbs_sign_messages

        try:
            # 1. Generate nonce
            nonce = os.urandom(16).hex()

            # 2. Create message to sign
            message = sha256(f"{cid_manifest}||{nonce}".encode())

            # 3. Sign with BBS+ to prove membership
            signature = bbs_sign_messages([message], self.bbs_secret_key)

            # 4. Request key from KRS
            Kc = krs.request_key(cid_manifest, nonce, signature)

            if Kc:
                logger.info(f" Node {self.node_id}: Received key from KRS")
                return Kc
            else:
                logger.error(f" Node {self.node_id}: KRS denied request")
                return None

        except Exception as e:
            logger.error(f"Node {self.node_id}: KRS request failed: {e}")
            return None
        
    async def aggregate_from_ipfs_with_krs(
        self, 
        ipfs_client, 
        blockchain_proxy, 
        auction_address, 
        krs,  # NUOVO: KeyReleaseService
        aggregation_method='fedavg'
    ):
        """
        Aggregatore scarica modelli da IPFS richiedendo chiavi da KRS.
        """
        logger.info(f"Node {self.node_id} (AGGREGATOR) aggregating with KRS")
        logger.info(f"Aggregation method: {aggregation_method.upper()}")
        
        try:
            # 1. Get hashes from blockchain
            import brownie
            contracts = brownie.project.chainServer
            auction_contract = contracts.AggregatorAuction.at(auction_address)
            
            ipfs_hashes = auction_contract.getAllModelHashes()
            logger.info(f"Found {len(ipfs_hashes)} model hashes on blockchain")
            
            # 2. Download models using KRS for keys
            downloaded_models = []
            
            for idx, ipfs_hash in enumerate(ipfs_hashes):
                if not ipfs_hash:
                    continue
                
                node_index = int(self.node_id)
                if idx == node_index:
                    # Use own model
                    downloaded_models.append(self.get_model_state_dict())
                    logger.info(f"Using own model")
                else:
                    # NUOVO: Request key from KRS
                    logger.info(f"Requesting key from KRS for model {idx}")
                    
                    Kc = self.request_key_from_krs(krs, ipfs_hash)
                    
                    if not Kc:
                        logger.warning(f"Failed to get key for model {idx}")
                        continue
                    
                    # Download with key
                    try:
                        model_data = ipfs_client.download_model_secured(ipfs_hash, Kc)
                        
                        if model_data:
                            downloaded_models.append(model_data)
                            logger.info(f"✓ Downloaded and decrypted model {idx}")
                        else:
                            logger.warning(f"Failed to download model {idx}")
                            
                    except Exception as e:
                        logger.error(f"Error downloading model {idx}: {e}")
                        continue
                    
            logger.info(f"Downloaded {len(downloaded_models)} models")
            
            # 3. Aggregate (existing code)
            if aggregation_method.lower() == 'krum':
                logger.info("Using KRUM aggregation")
                from server.aggregation_alg.krum import krumAggregator
                aggregator = krumAggregator(byzantine_ratio=0.3)
                aggregated_state = aggregator._aggregate_alg(downloaded_models)
                
            elif aggregation_method.lower() == 'median':
                logger.info("Using MEDIAN aggregation")
                from server.aggregation_alg.median import medianAggregator
                aggregator = medianAggregator()
                aggregated_state = aggregator._aggregate_alg(downloaded_models)
                
            else:
                logger.info("Using FEDAVG aggregation")
                aggregated_state = {}
                num_models = len(downloaded_models)
                
                for key in downloaded_models[0].keys():
                    aggregated_state[key] = sum(
                        model[key] for model in downloaded_models
                    ) / num_models
            
            logger.info(f"✓ Aggregation complete")
            
            return aggregated_state
            
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
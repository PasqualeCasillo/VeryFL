# protocols/AuctionProtocol.py
import logging
import asyncio
from typing import List, Dict, Any, Optional
from chainfl.krs import KeyReleaseService
from chainfl.revocation import VLRManager

from networkx import nodes
import torch
from node.DecentralizedNode import DecentralizedNode

from server.aggregation_alg.krum import krumAggregator
from server.aggregation_alg.median import medianAggregator

import os
from chainfl.crypto_bbs import bbs_generate_keypair

logger = logging.getLogger(__name__)

class AuctionProtocol:
    def __init__(self, blockchain_proxy, timeout_seconds: int = 300, 
                 aggregation_method: str = 'fedavg', attack_config=None):
        self.blockchain = blockchain_proxy
        self.timeout_seconds = timeout_seconds
        self.current_auction_address = None
        self.aggregation_method = aggregation_method
        self.attack_config = attack_config
        # NUOVO: KRS e VLR
        
        # Inizializza aggregatore
        if aggregation_method == 'krum':
            self.aggregator = krumAggregator()
        elif aggregation_method == 'median':
            self.aggregator = medianAggregator()
        else:
            self.aggregator = None  # FedAvg default
        
        # NUOVO: Gestione chiavi gruppo BBS+
        self.group_public_keys = []
        self.node_keypairs = {}  # {node_id: (public_key, secret_key)}
        
        self.krs = None
        self.vlr_manager = VLRManager()
        
    async def execute_round(self, round_num: int, nodes: List[DecentralizedNode]) -> Optional[dict]:
        """Execute a complete auction-based FL round"""
        try:
            logger.info(f"Round {round_num + 1}")
            active_nodes = [
                node for node in nodes 
                if not self.vlr_manager.is_revoked(node.node_id)
            ]
            
            revoked_count = len(nodes) - len(active_nodes)
            if revoked_count > 0:
                logger.warning(f"  {revoked_count} nodes excluded (revoked)")
                logger.warning(f"Revoked nodes: {self.vlr_manager.get_revoked()}")
            if round_num == 0:
                self.setup_group_keys(active_nodes)
            
                from chainfl.ipfs_client import IPFSClient
                ipfs_client = IPFSClient()
                ipfs_client.set_group_keys(self.group_public_keys)
                ipfs_client.enable_encryption(True)
                logger.info("IPFS client configured with encryption")
        
            # VERIFICA: KRS deve essere inizializzato
            if self.krs is None:
                logger.error("CRITICAL: KRS not initialized!")
                return None
            
            # Phase 1: Deploy auction contract
            auction_address = await self._deploy_auction_contract(round_num, nodes)
            if not auction_address:
                logger.error(f"Failed to deploy auction contract for round {round_num}")
                return None
            logger.info(f"Auction deployed")
                
            # Phase 2: Collect offers from nodes
            success = await self._collect_offers(nodes, auction_address)
            if not success:
                logger.warning(f"Not all nodes submitted offers for round {round_num}")
            logger.info(f"Collected {sum(1 for _ in nodes)} offers")
                
            # Phase 3: Wait for auction to close and get elected aggregator
            elected_aggregator = await self._wait_for_election(auction_address)
            if not elected_aggregator:
                logger.error(f"No aggregator elected for round {round_num}")
                return None
            logger.info(f"Aggregator: {elected_aggregator[:10]}...")
                
            # logger.info(f"Aggregator elected for round {round_num}: {elected_aggregator}")
            
            # Phase 4: Execute FL round with elected aggregator
            fl_result = await self._execute_fl_round(nodes, elected_aggregator, round_num)
            
            if not fl_result:
                return None
            
            # Phase 5: Calculate aggregate loss
            aggregate_loss = self._calculate_aggregate_loss(nodes)
            logger.info(f"Round complete: loss={aggregate_loss:.4f}")
            
            return {
                'success': True,
                'round': round_num,
                'aggregator': elected_aggregator,
                'aggregate_loss': aggregate_loss
            }
            
        except Exception as e:
            logger.error(f"Error in auction protocol round {round_num}: {e}")
            return None
            
    async def _deploy_auction_contract(self, round_num: int, nodes: List[DecentralizedNode]) -> Optional[str]:
        """Deploy auction contract for the round"""
        try:
            # MODIFICA: Usa account Ganache reali invece di indirizzi mock
            import brownie
            
            # Prendi i primi N account da Ganache (1-indexed perché 0 è il deployer)
            node_addresses = [brownie.accounts[i+1].address for i in range(len(nodes))]
            
            logger.info(f"Using real Ganache addresses: {node_addresses}")
            
            # Deploy contract via blockchain proxy
            auction_address = self.blockchain.deploy_auction_contract(
                whitelist=node_addresses,
                timeout_seconds=self.timeout_seconds,
                round_number=round_num
            )
            
            self.current_auction_address = auction_address
            logger.info(f"Deployed auction contract at {auction_address} for round {round_num}")
            return auction_address
            
        except Exception as e:
            logger.error(f"Failed to deploy auction contract: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
            
    async def _collect_offers(self, nodes: List[DecentralizedNode], auction_address: str) -> bool:
        """Collect offers from all nodes"""
        offer_tasks = []
        
        for node in nodes:
            task = asyncio.create_task(self._submit_node_offer(node, auction_address))
            offer_tasks.append(task)
            
        # Wait for all offers with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*offer_tasks, return_exceptions=True),
                timeout=self.timeout_seconds
            )
            
            successful_offers = sum(1 for result in results if result is True)
            logger.info(f"Collected {successful_offers}/{len(nodes)} offers")
            
            return successful_offers > 0
            
        except asyncio.TimeoutError:
            logger.warning("Timeout while collecting offers")
            return False
            
    async def _submit_node_offer(self, node: DecentralizedNode, auction_address: str) -> bool:
        """Submit offer for a single node"""
        try:
            # MODIFICA: Usa account Ganache reale
            import brownie
            node_index = int(node.node_id)
            real_address = brownie.accounts[node_index].address
            
            offer = node.get_auction_offer()
            
            success = self.blockchain.submit_offer(
                auction_address=auction_address,
                node_address=real_address,  # Usa indirizzo reale
                **offer
            )
            
            if success:
                logger.info(f"Node {node.node_id} submitted offer successfully")
            else:
                logger.warning(f"Node {node.node_id} failed to submit offer")
                
            return success
            
        except Exception as e:
            logger.error(f"Error submitting offer for node {node.node_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    async def _wait_for_election(self, auction_address: str) -> Optional[str]:
        max_wait_time = self.timeout_seconds + 60
        check_interval = 10
        checks = max_wait_time // check_interval
        
        for attempt in range(checks):
            try:
                election_result = self.blockchain.get_election_result(auction_address)
                if election_result:
                    return election_result
            except Exception as e:
                logger.warning(f"Election check attempt {attempt+1}/{checks} failed: {e}")
            
            if attempt < checks - 1:  # Non aspettare dopo l'ultimo tentativo
                await asyncio.sleep(check_interval)
        
        logger.error(f"Election timeout after {max_wait_time}s")
        return None
        

    async def _execute_fl_round(self, nodes, elected_aggregator, round_num):
        import brownie
        from chainfl.ipfs_client import IPFSClient
        import traceback

        ipfs_client = IPFSClient()
        ipfs_client.set_group_keys(self.group_public_keys)
        ipfs_client.enable_encryption(True)

        # 1. Assign roles
        aggregator_node = None
        for node in nodes:
            node_index = int(node.node_id)
            real_address = brownie.accounts[node_index].address

            if real_address.lower() == elected_aggregator.lower():
                node.set_role("aggregator", round_num)
                aggregator_node = node
            else:
                node.set_role("participant", round_num)

        if not aggregator_node:
            logger.error("Aggregator node not found in node list")
            return None

        # 2. Parallel training
        logger.info("Starting parallel training for all nodes...")
        await self._train_all_nodes(nodes)

        # 3. Participants upload to IPFS with encryption
        logger.info("Participants uploading encrypted models to IPFS...")
        upload_tasks = []
        for node in nodes:
            if node.role == "participant":
                task = node.upload_model_to_blockchain_and_ipfs(
                    ipfs_client, 
                    self.blockchain, 
                    self.current_auction_address
                )
                upload_tasks.append(asyncio.create_task(task))

        results = await asyncio.gather(*upload_tasks)
        successful_uploads = sum(results)
        logger.info(f"{successful_uploads}/{len(upload_tasks)} participants uploaded successfully")

        # 4. KEY SHARING PHASE - Participants share keys with aggregator
        logger.info("=" * 60)
        logger.info("KEY DISTRIBUTION PHASE - Registering keys in KRS")
        logger.info("=" * 60)

        if self.krs is None:
            logger.error("CRITICAL: KRS not initialized! Cannot register keys.")
            return None

        registered_count = 0
        for node in nodes:
            if node.role == "participant" and hasattr(node, '_encryption_keys'):
                for cid_manifest, Kc_b64 in node._encryption_keys.items():
                    self.krs.register_key(cid_manifest, Kc_b64)
                    registered_count += 1
                    logger.debug(f"Registered key for node {node.node_id}")

        logger.info(f" {registered_count} keys registered in KRS")
        logger.info("=" * 60)

        # 5. Verify upload completeness
        all_uploaded, missing_nodes = self.blockchain.verify_all_models_uploaded(
            self.current_auction_address
        )

        if not all_uploaded:
            logger.error(f"Upload verification failed: {len(missing_nodes)} nodes missing")
            logger.error(f"Missing node addresses: {missing_nodes}")

            remaining_nodes = len(nodes) - len(missing_nodes)

            if self.aggregation_method == 'krum':
                f = int(remaining_nodes * self.attack_config.byzantine_ratio)
                required = 2 * f + 3

                logger.error("=" * 60)
                logger.error("KRUM CONSTRAINT CHECK WITH MISSING UPLOADS")
                logger.error("=" * 60)
                logger.error(f"Byzantine ratio: {self.attack_config.byzantine_ratio}")
                logger.error(f"Remaining nodes: {remaining_nodes}")
                logger.error(f"Assumed Byzantine (f): {f}")
                logger.error(f"Required nodes for Krum: {required}")
                logger.error(f"Constraint satisfied: {remaining_nodes >= required}")
                logger.error("=" * 60)

                if remaining_nodes < required:
                    logger.error(
                        f"ABORTING ROUND: Krum constraint violated "
                        f"(available: {remaining_nodes}, required: {required})"
                    )
                    logger.error("System cannot guarantee Byzantine-fault-tolerance with current node set")
                    return None
                else:
                    logger.warning(
                        f"Continuing with reduced node set "
                        f"({remaining_nodes} nodes available, threshold: {required})"
                    )
            else:
                logger.error("Aborting round due to incomplete uploads")
                return None

        logger.info("Upload verification passed: all required models available")

        # 6. Aggregator downloads, aggregates and uploads to IPFS
        logger.info("=" * 60)
        logger.info("AGGREGATION PHASE - Aggregator requesting keys from KRS")
        logger.info(f"Aggregation method: {self.aggregation_method.upper()}")
        logger.info(f"Aggregator node: {aggregator_node.node_id}")

        try:
            aggregated_model = await aggregator_node.aggregate_from_ipfs_with_krs(
                ipfs_client,
                self.blockchain,
                self.current_auction_address,
                self.krs,
                aggregation_method=self.aggregation_method
            )

            if not aggregated_model:
                logger.error("Aggregation returned None - operation failed")
                return None

            logger.info("Aggregation completed successfully")

        except ValueError as e:
            logger.error("=" * 60)
            logger.error("AGGREGATION FAILED: CONSTRAINT VIOLATION")
            logger.error("=" * 60)
            logger.error(f"Error: {str(e)}")
            logger.error("=" * 60)
            logger.error("Aborting round due to aggregation failure")
            return None

        except Exception as e:
            logger.error("=" * 60)
            logger.error("AGGREGATION FAILED: UNEXPECTED ERROR")
            logger.error("=" * 60)
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error("Stack trace:")
            logger.error(traceback.format_exc())
            logger.error("=" * 60)
            return None

        # 7. Upload global model to IPFS with encryption
        logger.info("Uploading global aggregated model to IPFS...")
        global_ipfs_hash = await aggregator_node.upload_global_model_to_ipfs(
            ipfs_client,
            aggregated_model,
            self.current_auction_address
        )

        if not global_ipfs_hash:
            logger.error("Global model upload to IPFS failed")
            return None

        logger.info(f"Global model uploaded to IPFS: {global_ipfs_hash}")

        # # 8. KEY SHARING PHASE - Aggregator shares global model key with all nodes
        # logger.info("Distributing global model encryption key to all nodes...")
        # if hasattr(aggregator_node, '_encryption_keys') and global_ipfs_hash in aggregator_node._encryption_keys:
        #     global_Kc = aggregator_node._encryption_keys[global_ipfs_hash]
            
        #     for node in nodes:
        #         if node.node_id != aggregator_node.node_id:
        #             node.store_encryption_key(global_ipfs_hash, global_Kc)
            
        #     logger.info("Global model key distributed to all nodes")
        # else:
        #     logger.warning("No encryption key found for global model")
        
        # 8. KEY SHARING PHASE - Register key in KRS for secure distribution
        logger.info("Registering global model key in KRS...")
        if hasattr(aggregator_node, '_encryption_keys') and global_ipfs_hash in aggregator_node._encryption_keys:
            global_Kc = aggregator_node._encryption_keys[global_ipfs_hash]

            # OPZIONE 2: Registra nel KRS (Produzione)
            self.krs.register_key(global_ipfs_hash, global_Kc)
            logger.info(f" Global model key registered in KRS for CID {global_ipfs_hash[:10]}...")

            # OPZIONE 2: Notifica ai nodi che possono richiedere la chiave
            logger.info("Nodes will request key from KRS during download phase")

        else:
            logger.warning("No encryption key found for global model")

        # 9. Decentralized distribution via IPFS
        logger.info("Initiating decentralized model distribution...")
        distribution_success = await self._distribute_via_ipfs(nodes, ipfs_client)

        if not distribution_success:
            logger.warning("Some nodes failed to download global model")

        logger.info("=" * 60)
        logger.info("FL ROUND COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

        return aggregated_model
    
    async def _upload_node_model_secured(self, node: DecentralizedNode, ipfs_client, 
                                         blockchain_proxy, auction_address) -> bool:
        """
        Upload node model with encryption and group signature
        
        Args:
            node: DecentralizedNode instance
            ipfs_client: IPFSClient with encryption enabled
            blockchain_proxy: Blockchain proxy for registration
            auction_address: Current auction contract address
        
        Returns:
            True if upload successful, False otherwise
        """
        try:
            logger.info(f"Node {node.node_id} uploading encrypted model...")
            
            # Get model data
            model_state_dict = node.get_model_state_dict()
            metadata = {
                'node_id': node.node_id,
                'round': node.current_round,
                'role': node.role
            }
            
            # Upload with encryption and signature
            cid_manifest, key_data = ipfs_client.upload_model_secured(
                model_state_dict,
                metadata,
                node.bbs_secret_key
            )
            
            if not cid_manifest or not key_data:
                logger.error(f"Node {node.node_id} failed IPFS upload")
                return False
            
            logger.info(f"Node {node.node_id} uploaded to IPFS: {cid_manifest}")
            
            # Register hash on blockchain
            import brownie
            node_index = int(node.node_id)
            node_account = brownie.accounts[node_index]
            
            contracts = brownie.project.chainServer
            auction_contract = contracts.AggregatorAuction.at(auction_address)
            
            tx = auction_contract.submitModelHash(cid_manifest, {'from': node_account})
            tx.wait(1)
            
            logger.info(f"Node {node.node_id} registered hash on blockchain")
            
            # Store encryption key for this node (in-memory for PoC)
            if not hasattr(node, '_encryption_keys'):
                node._encryption_keys = {}
            node._encryption_keys[cid_manifest] = key_data['Kc']
            
            return True
            
        except Exception as e:
            logger.error(f"Node {node.node_id} upload failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    #    NUOVA FUNZIONE: Distribuzione decentralizzata
    async def _distribute_via_ipfs(self, nodes: List[DecentralizedNode], ipfs_client):
        """
        I nodi scaricano AUTONOMAMENTE da IPFS leggendo l'hash dalla blockchain.
        Nessun trasferimento in-memory!
        """
        logger.info(" Nodes downloading global model from IPFS autonomously...")
        
        # Aspetta che tutti i nodi scarichino in PARALLELO
        download_tasks = [
            node.download_global_model_from_ipfs(
                ipfs_client, 
                self.current_auction_address,
                self.krs  # ← NUOVO: Passa riferimento KRS
            )
            for node in nodes
        ]
        
        results = await asyncio.gather(*download_tasks)
        successful = sum(results)
        
        logger.info(f" {successful}/{len(nodes)} nodes downloaded global model")
        logger.info(f" All key requests processed by KRS")
        return successful == len(nodes)
        
    async def _train_all_nodes(self, nodes: List[DecentralizedNode]) -> List[Dict]:
        """Train all nodes with metrics aggregation"""
        training_tasks = []
        for node in nodes:
            task = asyncio.create_task(self._train_node(node))
            training_tasks.append(task)
            
        training_results = await asyncio.gather(*training_tasks, return_exceptions=True)
        
        # Filter successful results
        successful_results = [r for r in training_results if isinstance(r, dict) and 'error' not in r]
        
        # Aggregate and log metrics
        if successful_results:
            avg_initial_loss = sum(r['training']['initial_loss'] for r in successful_results) / len(successful_results)
            avg_final_loss = sum(r['training']['final_loss'] for r in successful_results) / len(successful_results)
            avg_final_acc = sum(r['training']['final_accuracy'] for r in successful_results) / len(successful_results)
            
            logger.info(f"  Training: loss {avg_initial_loss:.3f}→{avg_final_loss:.3f}, acc {avg_final_acc:.1f}%")
        
        return training_results
        
    async def _train_node(self, node: DecentralizedNode) -> Dict[str, Any]:
        """Train a single node"""
        try:
            training_result = node.train_local_model()
            test_result = node.test_model()
            
            return {
                'training': training_result,
                'test': test_result,
                'node_id': node.node_id
            }
        except Exception as e:
            logger.error(f"Error training node {node.node_id}: {e}")
            return {'error': str(e), 'node_id': node.node_id}
        
        
    async def _upload_node_model(self, node: DecentralizedNode) -> str:
        """Upload singolo modello"""
        upload_params = {
            'epoch': node.current_round,
            'state_dict': node.get_model_state_dict(),
            'client_id': node.node_id,
            'timestamp': None
        }
        result = self.blockchain.upload_model(upload_params)
        logger.info(f"Node {node.node_id} model upload result: {result}")
        return result
            
    async def _upload_models(self, nodes: List[DecentralizedNode]) -> bool:
        try:
            upload_tasks = [self._upload_node_model(node) for node in nodes]
            results = await asyncio.gather(*upload_tasks, return_exceptions=True)

            successful = sum(1 for r in results if not isinstance(r, Exception))
            logger.info(f"Uploaded {successful}/{len(nodes)} models successfully")
            return successful > 0
        except Exception as e:
            logger.error(f"Error uploading models: {e}")
            return False
    
            
    async def _perform_aggregation(self, nodes, aggregator_address):
        """
        Delegate aggregation to the elected aggregator node.
        The aggregator collects models and computes aggregation using configured method.
        """
        try:
            import brownie

            # Find the aggregator node
            aggregator_node = None
            participant_nodes = []

            for node in nodes:
                node_index = int(node.node_id)
                real_address = brownie.accounts[node_index].address

                if real_address.lower() == aggregator_address.lower():
                    aggregator_node = node
                else:
                    participant_nodes.append(node)

            if not aggregator_node:
                logger.error("Aggregator node not found")
                return None

            logger.info(f"Aggregator: Node {aggregator_node.node_id}")
            logger.info(f"Participants: {len(participant_nodes)} nodes")
            logger.info(f"Aggregation method: {self.aggregation_method.upper()}")  # ← NUOVO LOG

            # Collect participant models (in-memory, no IPFS needed)
            participant_models = [
                node.get_model_state_dict() 
                for node in participant_nodes
            ]

            # KEY CHANGE: Passa il metodo di aggregazione
            aggregated_state = aggregator_node.aggregate_models(
                participant_models,
                method=self.aggregation_method  # ← AGGIUNGI QUESTO
            )
            
                        # AGGIUNGI ANALISI POST-AGGREGAZIONE
            if self.aggregation_method == 'krum':
                logger.info("=" * 60)
                logger.info("POST-AGGREGATION ANALYSIS")
                logger.info("=" * 60)
                
                # Flatten aggregato
                agg_flat = torch.cat([
                    aggregated_state[k].flatten() 
                    for k in sorted(aggregated_state.keys())
                ])
                
                # Calcola distanza aggregato da ogni modello originale
                all_models = participant_models + [aggregator_node.get_model_state_dict()]
                
                distances_from_agg = []
                for i, model in enumerate(all_models):
                    model_flat = torch.cat([model[k].flatten() for k in sorted(model.keys())])
                    dist = torch.norm(agg_flat - model_flat, p=2)
                    distances_from_agg.append(dist.item())
                    
                    # Identifica se nodo è Byzantine
                    node = nodes[i]
                    byz_marker = " [BYZANTINE]" if node.is_byzantine else " [HONEST]"
                    
                    logger.info(f"  Distance aggregated → node {i}{byz_marker}: {dist:.2e}")
                
                # Trova modello più vicino all'aggregato
                import numpy as np
                closest_idx = np.argmin(distances_from_agg)
                closest_node = nodes[closest_idx]
                
                logger.info(f"Aggregated model closest to: Node {closest_idx}")
                
                if closest_node.is_byzantine:
                    logger.error(f"✗ WARNING: Aggregated model is closest to a BYZANTINE node!")
                else:
                    logger.info(f" Aggregated model is closest to an HONEST node")
                
                logger.info("=" * 60)

            logger.info(f"{self.aggregation_method.upper()} aggregation completed by node {aggregator_node.node_id}")
            return aggregated_state

        except Exception as e:
            logger.error(f"Error in aggregation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
            
    # async def _distribute_global_model(self, nodes: List[DecentralizedNode], 
    #                                  global_model: Dict) -> bool:
    #     """Distribute global model to all nodes"""
    #     try:
    #         for node in nodes:
    #             node.load_state_dict(global_model)
                
    #         logger.info("Global model distributed to all nodes")
    #         return True
    #     except Exception as e:
    #         logger.error(f"Error distributing global model: {e}")
    #         return False
    
    def setup_group_keys(self, nodes: List[DecentralizedNode]):
        """Setup keys + initialize KRS"""
        logger.info("=" * 60)
        logger.info("SETTING UP BBS+ GROUP SIGNATURE KEYS + KRS")
        logger.info("=" * 60)
        
        self.group_public_keys = []
        self.node_keypairs = {}
        
        for node in nodes:
            seed = os.urandom(32)
            public_key, secret_key = bbs_generate_keypair(seed)
            
            self.node_keypairs[node.node_id] = (public_key, secret_key)
            self.group_public_keys.append(public_key)
            
            node.bbs_secret_key = secret_key
            
            logger.info(f"Generated BBS+ keypair for Node {node.node_id}")
        
        # NUOVO: Initialize KRS
        self.krs = KeyReleaseService(self.group_public_keys)
        
        logger.info(f"Total group members: {len(self.group_public_keys)}")
        logger.info("🔑 KRS initialized")
        logger.info("=" * 60)
        
        return self.group_public_keys
        
    # def setup_group_keys(self, nodes: List[DecentralizedNode]):
    #     """
    #     Generate and distribute BBS+ group signature keys to nodes
        
    #     Args:
    #         nodes: List of DecentralizedNode instances
    #     """
    #     logger.info("=" * 60)
    #     logger.info("SETTING UP BBS+ GROUP SIGNATURE KEYS")
    #     logger.info("=" * 60)
        
    #     self.group_public_keys = []
    #     self.node_keypairs = {}
        
    #     for node in nodes:
    #         # Generate keypair for this node
    #         seed = os.urandom(32)
    #         public_key, secret_key = bbs_generate_keypair(seed)
            
    #         # Store keypair
    #         self.node_keypairs[node.node_id] = (public_key, secret_key)
    #         self.group_public_keys.append(public_key)
            
    #         # Assign secret key to node
    #         node.bbs_secret_key = secret_key
            
    #         logger.info(f"Generated BBS+ keypair for Node {node.node_id}")
        
    #     logger.info(f"Total group members: {len(self.group_public_keys)}")
    #     logger.info("=" * 60)
        
    #     return self.group_public_keys
    
    def _calculate_aggregate_loss(self, nodes):
        """Calculate average loss from all nodes' training results"""
        total_loss = 0.0
        count = 0

        for node in nodes:
            if hasattr(node, '_last_training_loss') and node._last_training_loss is not None:
                total_loss += node._last_training_loss
                count += 1

        return total_loss / count if count > 0 else 0.0
    
    def configure_nodes_attack(self, nodes):
        """Configura nodi Byzantine secondo attack_config"""
        if not self.attack_config:
            logger.info("No attack configuration, all nodes honest")
            return
        
        total_nodes = len(nodes)
        for node in nodes:
            node.configure_attack(self.attack_config, total_nodes)
        
        byzantine_count = sum(1 for n in nodes if n.is_byzantine)
        logger.info(f"Attack configured: {byzantine_count}/{total_nodes} Byzantine nodes")
        
    def _quick_accuracy(self, model, dataloader):
        """Quick accuracy calculation without full metrics"""
        model.eval()
        correct = 0
        total = 0
    
        with torch.no_grad():
            for data, target in dataloader:
                output = model(data)
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
    
        return correct / total if total > 0 else 0.0

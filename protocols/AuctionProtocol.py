# protocols/AuctionProtocol.py
import logging
import asyncio
from typing import List, Dict, Any, Optional

from networkx import nodes
from node.DecentralizedNode import DecentralizedNode

from server.aggregation_alg.krum import krumAggregator
from server.aggregation_alg.median import medianAggregator

logger = logging.getLogger(__name__)

class AuctionProtocol:
    def __init__(self, blockchain_proxy, timeout_seconds: int = 300, 
                 aggregation_method: str = 'fedavg'):
        self.blockchain = blockchain_proxy
        self.timeout_seconds = timeout_seconds
        self.current_auction_address = None
        self.aggregation_method = aggregation_method
        
        # Inizializza aggregatore
        if aggregation_method == 'krum':
            self.aggregator = krumAggregator()
        elif aggregation_method == 'median':
            self.aggregator = medianAggregator()
        else:
            self.aggregator = None  # FedAvg default
        
    async def execute_round(self, round_num: int, nodes: List[DecentralizedNode]) -> Optional[dict]:
        """Execute a complete auction-based FL round"""
        try:
            logger.info(f"Round {round_num + 1}")            
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
            logger.info(f"✓ Aggregator: {elected_aggregator[:10]}...")
                
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
        
    async def _execute_fl_round(self, nodes: List[DecentralizedNode], 
                              elected_aggregator: str, round_num: int) -> str:
        """Execute FL training round with elected aggregator"""
        # MODIFICA: Confronta con indirizzi reali Ganache
        import brownie
        
        for node in nodes:
            node_index = int(node.node_id)
            real_address = brownie.accounts[node_index].address
            
            if real_address.lower() == elected_aggregator.lower():
                node.set_role("aggregator", round_num)
            else:
                node.set_role("participant", round_num)
                
        # Execute parallel training with metrics collection
        training_results = await self._train_all_nodes(nodes)
        logger.info(f"Completed training for {len(training_results)} nodes")
        
        # Upload models and perform aggregation
        upload_results = await self._upload_models(nodes)
        if upload_results:
            aggregated_model = await self._perform_aggregation(nodes, elected_aggregator)
            if aggregated_model:
                await self._distribute_global_model(nodes, aggregated_model)
                return aggregated_model
                
        return None
        
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
    
            
    # protocols/AuctionProtocol.py - metodo _perform_aggregation
    async def _perform_aggregation(self, nodes: List[DecentralizedNode], 
                                     aggregator_address: str) -> Optional[Dict]:
            try:
                import brownie
                
                aggregator_node = None
                for node in nodes:
                    node_index = int(node.node_id)
                    real_address = brownie.accounts[node_index].address
                    if real_address.lower() == aggregator_address.lower():
                        aggregator_node = node
                        break
                        
                if not aggregator_node:
                    logger.error("Aggregator node not found")
                    return None
                    
                model_states = [node.get_model_state_dict() for node in nodes]
                
                # Verifica compatibilità
                first_keys = set(model_states[0].keys())
                for i, state in enumerate(model_states[1:], 1):
                    if set(state.keys()) != first_keys:
                        logger.error(f"Node {i} model keys mismatch")
                        return None
                    for key in first_keys:
                        if model_states[0][key].shape != state[key].shape:
                            logger.error(f"Node {i} tensor shape mismatch for {key}")
                            return None
                
                # Scegli metodo aggregazione
                if self.aggregator:
                    logger.info(f"Using {self.aggregation_method} aggregation")
                    aggregated_state = self.aggregator._aggregate_alg(model_states)
                else:
                    # FedAvg default
                    logger.info("Using FedAvg aggregation")
                    aggregated_state = {}
                    num_models = len(model_states)
                    for key in model_states[0].keys():
                        aggregated_state[key] = sum(state[key] for state in model_states) / num_models
                        
                logger.info(f"Aggregation completed by node {aggregator_node.node_id}")
                return aggregated_state
                
            except Exception as e:
                logger.error(f"Error in aggregation: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return None
            
    async def _distribute_global_model(self, nodes: List[DecentralizedNode], 
                                     global_model: Dict) -> bool:
        """Distribute global model to all nodes"""
        try:
            for node in nodes:
                node.load_state_dict(global_model)
                
            logger.info("Global model distributed to all nodes")
            return True
        except Exception as e:
            logger.error(f"Error distributing global model: {e}")
            return False
        
    def _calculate_aggregate_loss(self, nodes):
        """Calculate average loss from all nodes' training results"""
        total_loss = 0.0
        count = 0

        for node in nodes:
            if hasattr(node, '_last_training_loss') and node._last_training_loss is not None:
                total_loss += node._last_training_loss
                count += 1

        return total_loss / count if count > 0 else 0.0
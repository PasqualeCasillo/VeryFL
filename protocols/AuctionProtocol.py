# protocols/AuctionProtocol.py
import logging
import asyncio
from typing import List, Dict, Any, Optional
from node.DecentralizedNode import DecentralizedNode

logger = logging.getLogger(__name__)

class AuctionProtocol:
    def __init__(self, blockchain_proxy, timeout_seconds: int = 300):
        self.blockchain = blockchain_proxy
        self.timeout_seconds = timeout_seconds
        self.current_auction_address = None
        
    async def execute_round(self, round_num: int, nodes: List[DecentralizedNode]) -> Optional[str]:
        """Execute a complete auction-based FL round"""
        try:
            logger.info(f"Starting auction protocol for round {round_num}")
            
            # Phase 1: Deploy auction contract
            auction_address = await self._deploy_auction_contract(round_num, nodes)
            if not auction_address:
                logger.error(f"Failed to deploy auction contract for round {round_num}")
                return None
                
            # Phase 2: Collect offers from nodes
            success = await self._collect_offers(nodes, auction_address)
            if not success:
                logger.warning(f"Not all nodes submitted offers for round {round_num}")
                
            # Phase 3: Wait for auction to close and get elected aggregator
            elected_aggregator = await self._wait_for_election(auction_address)
            if not elected_aggregator:
                logger.error(f"No aggregator elected for round {round_num}")
                return None
                
            logger.info(f"Aggregator elected for round {round_num}: {elected_aggregator}")
            
            # Phase 4: Execute FL round with elected aggregator
            return await self._execute_fl_round(nodes, elected_aggregator, round_num)
            
        except Exception as e:
            logger.error(f"Error in auction protocol round {round_num}: {e}")
            return None
            
    async def _deploy_auction_contract(self, round_num: int, nodes: List[DecentralizedNode]) -> Optional[str]:
        """Deploy auction contract for the round"""
        try:
            # Get node addresses (simplified - in practice would use actual blockchain addresses)
            node_addresses = [f"0x{i:040d}" for i in range(len(nodes))]
            
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
            offer = node.get_auction_offer()
            
            success = self.blockchain.submit_offer(
                auction_address=auction_address,
                node_address=f"0x{int(node.node_id):040d}",
                **offer
            )
            
            if success:
                logger.info(f"Node {node.node_id} submitted offer successfully")
            else:
                logger.warning(f"Node {node.node_id} failed to submit offer")
                
            return success
            
        except Exception as e:
            logger.error(f"Error submitting offer for node {node.node_id}: {e}")
            return False
            
    async def _wait_for_election(self, auction_address: str) -> Optional[str]:
        """Wait for auction to close and return elected aggregator"""
        max_wait_time = self.timeout_seconds + 60  # Extra buffer
        check_interval = 10  # seconds
        
        for _ in range(max_wait_time // check_interval):
            try:
                election_result = self.blockchain.get_election_result(auction_address)
                if election_result:
                    return election_result
                    
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error checking election result: {e}")
                await asyncio.sleep(check_interval)
                
        logger.error("Timeout waiting for election result")
        return None
        
    async def _execute_fl_round(self, nodes: List[DecentralizedNode], 
                              elected_aggregator: str, round_num: int) -> str:
        """Execute FL training round with elected aggregator"""
        # Set roles
        for node in nodes:
            if f"0x{int(node.node_id):040d}" == elected_aggregator:
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
            
            logger.info(f"Round aggregate: loss {avg_initial_loss:.4f}→{avg_final_loss:.4f}, "
                       f"accuracy {avg_final_acc:.2f}%")
        
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
            
    async def _upload_models(self, nodes: List[DecentralizedNode]) -> bool:
        """Upload all node models to IPFS"""
        try:
            for node in nodes:
                upload_params = {
                    'epoch': node.current_round,
                    'state_dict': node.get_model_state_dict(),
                    'client_id': node.node_id,
                    'timestamp': None
                }
                
                result = self.blockchain.upload_model(upload_params)
                logger.info(f"Node {node.node_id} model upload result: {result}")
                
            return True
        except Exception as e:
            logger.error(f"Error uploading models: {e}")
            return False
            
    async def _perform_aggregation(self, nodes: List[DecentralizedNode], 
                                 aggregator_address: str) -> Optional[Dict]:
        """Perform model aggregation"""
        try:
            # Find aggregator node
            aggregator_node = None
            for node in nodes:
                if f"0x{int(node.node_id):040d}" == aggregator_address:
                    aggregator_node = node
                    break
                    
            if not aggregator_node:
                logger.error("Aggregator node not found")
                return None
                
            # Collect all model state dicts
            model_states = [node.get_model_state_dict() for node in nodes]
            
            # Simple FedAvg aggregation
            aggregated_state = {}
            num_models = len(model_states)
            
            for key in model_states[0].keys():
                aggregated_state[key] = sum(state[key] for state in model_states) / num_models
                
            logger.info(f"Aggregation completed by node {aggregator_node.node_id}")
            return aggregated_state
            
        except Exception as e:
            logger.error(f"Error in aggregation: {e}")
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
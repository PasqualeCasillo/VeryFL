# DecentralizedTask.py
import logging
import asyncio
from task import Task
from node.DecentralizedNode import DecentralizedNode
from protocols.AuctionProtocol import AuctionProtocol
from chainfl.auction_proxy import auction_chain_proxy
from copy import deepcopy

logger = logging.getLogger(__name__)

class DecentralizedTask(Task):
    def __init__(self, global_args, train_args, algorithm):
        super().__init__(global_args, train_args, algorithm)
        self.nodes = []
        self.auction_protocol = AuctionProtocol(
            blockchain_proxy=auction_chain_proxy,
            timeout_seconds=global_args.get('auction_timeout', 300)
        )
        
    def _construct_nodes(self):
        """Construct decentralized nodes instead of traditional clients"""
        logger.info(f"Constructing {len(self.client_list)} decentralized nodes")
        
        for client_id, _ in self.client_list.items():
            node = DecentralizedNode(
                node_id=client_id,
                model=deepcopy(self.model),
                dataloader=self.train_dataloader_list[client_id],
                trainer_class=self.trainer,
                train_args=self.train_args,
                test_dataloader=self.test_dataloader
            )
            self.nodes.append(node)
            
        logger.info(f"Created {len(self.nodes)} decentralized nodes")
        
    def run_decentralized(self):
        """Run decentralized federated learning with auction-based aggregator election"""
        logger.info("Starting decentralized federated learning with auction protocol")
        
        # Setup phase
        self._regist_client()
        self._construct_dataloader()
        self._construct_sign()
        self._construct_nodes()
        
        # Run auction-based FL rounds
        asyncio.run(self._run_auction_rounds())
        
    async def _run_auction_rounds(self):
        """Run FL rounds with auction protocol"""
        for round_num in range(self.global_args['communication_round']):
            logger.info(f"Starting auction-based FL round {round_num + 1}/{self.global_args['communication_round']}")
            
            try:
                result = await self.auction_protocol.execute_round(round_num, self.nodes)
                
                if result:
                    logger.info(f"Round {round_num + 1} completed successfully")
                else:
                    logger.warning(f"Round {round_num + 1} failed")
                    
            except Exception as e:
                logger.error(f"Critical error in round {round_num + 1}: {e}")
                
        logger.info("Decentralized federated learning completed")
        
    def run(self):
        """Run with choice between centralized and decentralized mode"""
        mode = self.global_args.get('mode', 'centralized')
        
        if mode == 'decentralized':
            self.run_decentralized()
        else:
            super().run()
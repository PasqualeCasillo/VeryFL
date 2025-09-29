# task/DecentralizedTask.py
import logging
import asyncio
from task import Task
from node.DecentralizedNode import DecentralizedNode
from protocols.AuctionProtocol import AuctionProtocol
from chainfl.auction_proxy import auction_chain_proxy
from utils.metrics import MetricsCalculator
from utils.metrics_logger import MetricsLogger
from utils.plotter import MetricsPlotter
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
        self.metrics_calculator = MetricsCalculator()
        self.metrics_logger = MetricsLogger(save_dir=global_args.get('results_dir', 'results'))
        self.plotter = MetricsPlotter(save_dir=global_args.get('results_dir', 'results') + '/plots')
        
    def _construct_nodes(self):
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
        logger.info("Starting decentralized federated learning with auction protocol")
        
        self._regist_client()
        self._construct_dataloader()
        self._construct_sign()
        self._construct_nodes()
        
        asyncio.run(self._run_auction_rounds())
        
        # Generate plots and save metrics
        self.metrics_logger.save()
        self.plotter.plot_all(self.metrics_logger.metrics)
        logger.info(f"Metrics and plots saved to {self.metrics_logger.save_dir}")
        
    async def _run_auction_rounds(self):
        for round_num in range(self.global_args['communication_round']):
            logger.info(f"Starting auction-based FL round {round_num + 1}/{self.global_args['communication_round']}")
            
            try:
                result = await self.auction_protocol.execute_round(round_num, self.nodes)
                
                if result:
                    # Calculate metrics
                    aggregator = next((n for n in self.nodes if n.role == 'aggregator'), self.nodes[0])
                    
                    global_metrics = self.metrics_calculator.calculate_all_metrics(
                        aggregator.model,
                        self.test_dataloader
                    )
                    
                    node_metrics = {}
                    for node in self.nodes:
                        node_metrics[node.node_id] = self.metrics_calculator.calculate_all_metrics(
                            node.model,
                            node.test_dataloader if node.test_dataloader else self.test_dataloader
                        )
                    
                    avg_loss = result.get('aggregate_loss', 0.0)
                    
                    self.metrics_logger.log_round(
                        round_num + 1,
                        global_metrics,
                        node_metrics,
                        avg_loss
                    )
                    
                    logger.info(
                        f"Round {round_num + 1}: Loss={avg_loss:.4f}, "
                        f"Acc={global_metrics['accuracy']:.4f}, "
                        f"F1={global_metrics['f1']:.4f}, "
                        f"Prec={global_metrics['precision']:.4f}, "
                        f"Rec={global_metrics['recall']:.4f}, "
                        f"AUC={global_metrics['auc']:.4f}"
                    )
                    logger.info(f"Round {round_num + 1} completed successfully")
                else:
                    logger.warning(f"Round {round_num + 1} failed")
                    
            except Exception as e:
                logger.error(f"Critical error in round {round_num + 1}: {e}")
                
        logger.info("Decentralized federated learning completed")
        
    def run(self):
        mode = self.global_args.get('mode', 'centralized')
        
        if mode == 'decentralized':
            self.run_decentralized()
        else:
            super().run()
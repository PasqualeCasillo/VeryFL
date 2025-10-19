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
from config.attack_config import AttackConfig

logger = logging.getLogger(__name__)

class DecentralizedTask(Task):
    def __init__(self, global_args, train_args, algorithm):
        super().__init__(global_args, train_args, algorithm)
        self.nodes = []
        
        aggregation_method = global_args.get('aggregation_method', 'fedavg')
        
        # Configurazione attacco
        self.attack_config = AttackConfig(
            attack_type=global_args.get('attack_type', 'none'),
            byzantine_ratio=global_args.get('byzantine_ratio', 0.0),
            attack_start_round=global_args.get('attack_start_round', 0)
        )

        self.auction_protocol = AuctionProtocol(
            blockchain_proxy=auction_chain_proxy,
            timeout_seconds=global_args.get('auction_timeout', 300),
            aggregation_method=aggregation_method,
            attack_config=self.attack_config
        )
        
        # NUOVO: Configurazione encryption
        self.encryption_enabled = global_args.get('enable_encryption', False)
        if self.encryption_enabled:
            logger.info("=" * 60)
            logger.info("ENCRYPTION ENABLED")
            logger.info("Models will be encrypted with AES-256-GCM")
            logger.info("Group signatures enabled for authentication")
            logger.info("=" * 60)
        
        # Usa save_dir unico per tutti i file
        save_dir = global_args.get('results_dir', 'results')
        self.metrics_logger = MetricsLogger(save_dir=save_dir)
        self.plotter = MetricsPlotter(save_dir=f'{save_dir}/plots')    
    def _construct_nodes(self):
        logger.info(f"Constructing {len(self.client_list)} decentralized nodes")
        
        num_classes = self.global_args.get('class_num', 10)
        
        for client_id, _ in self.client_list.items():
            node = DecentralizedNode(
                node_id=client_id,
                model=deepcopy(self.model),
                dataloader=self.train_dataloader_list[client_id],
                trainer_class=self.trainer,
                train_args=self.train_args,
                test_dataloader=self.test_dataloader,
                num_classes=num_classes
            )
            self.nodes.append(node)
        
        # Configura attacco sui nodi
        self.auction_protocol.configure_nodes_attack(self.nodes)
        
        logger.info(f"Created {len(self.nodes)} decentralized nodes")
        
    def run_decentralized(self):
        logger.info("Starting decentralized federated learning with auction protocol")
        
        self._regist_client()
        self._construct_dataloader()
        self._construct_sign()
        self._construct_nodes()
        
        asyncio.run(self._run_auction_rounds())
        
        # Salva con nome distintivo per dataset e modalità
        dataset_name = self.global_args.get('dataset', 'unknown')
        mode = self.global_args.get('mode', 'centralized')
        
        self.metrics_logger.save(f'{dataset_name}_{mode}_metrics.json')
        self.plotter.plot_all(self.metrics_logger.metrics)
        logger.info(f"Metrics and plots saved to {self.metrics_logger.save_dir}")
        
    async def _run_auction_rounds(self):
        for round_num in range(self.global_args['communication_round']):
            try:
                result = await self.auction_protocol.execute_round(round_num, self.nodes)
                
                if result:
                    aggregator = next((n for n in self.nodes if n.role == 'aggregator'), self.nodes[0])
                    
                    global_metrics = self.metrics_calculator.calculate_all_metrics(
                        aggregator.model,
                        self.test_dataloader
                    )
                    
                    # Per-node metrics
                    node_metrics = {}
                    for node in self.nodes:
                        # test_loader = self.test_dataloader
                        test_loader = node.dataloader
                        
                        
                        node_metrics[node.node_id] = self.metrics_calculator.calculate_all_metrics(
                            node.model,
                            test_loader,
                            device=self.train_args.get('device', 'cpu')
                        )
                        
                        # NUOVO: Aggiungi flag Byzantine alle metriche
                        node_metrics[node.node_id]['is_byzantine'] = node.is_byzantine
                    
                    avg_loss = result.get('aggregate_loss', 0.0)
                    
                    # NUOVO: Identifica nodi Byzantine e stato attacco
                    byzantine_nodes = [n.node_id for n in self.nodes if n.is_byzantine]
                    attack_active = self.attack_config.is_active(round_num)
                    
                    self.metrics_logger.log_round(
                        round_num + 1,
                        global_metrics,
                        node_metrics,
                        avg_loss,
                        byzantine_nodes=byzantine_nodes,  # NUOVO
                        attack_active=attack_active       # NUOVO
                    )
                    
                    # NUOVO: Log dettagliato per debug
                    if attack_active:
                        honest_acc = [node_metrics[n.node_id]['accuracy'] 
                                     for n in self.nodes if not n.is_byzantine]
                        byz_acc = [node_metrics[n.node_id]['accuracy'] 
                                  for n in self.nodes if n.is_byzantine]
                        
                        logger.info(
                            f"  Attack Active - "
                            f"Honest nodes avg acc: {sum(honest_acc)/len(honest_acc):.3f}, "
                            f"Byzantine nodes avg acc: {sum(byz_acc)/len(byz_acc) if byz_acc else 0:.3f}"
                        )
                    
                    logger.info(
                        f"  Global → Acc={global_metrics['accuracy']:.3f}, "
                        f"F1={global_metrics['f1']:.3f}, "
                        f"Prec={global_metrics['precision']:.3f}, "
                        f"Rec={global_metrics['recall']:.3f}"
                    )
                    
                else:
                    logger.warning(f"Round {round_num + 1} failed")
                    
            except Exception as e:
                logger.error(f"Critical error in round {round_num + 1}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info("Training Complete")
        logger.info(f"Results saved to: {self.metrics_logger.save_dir}")
        
    def run(self):
        mode = self.global_args.get('mode', 'centralized')
        
        if mode == 'decentralized':
            self.run_decentralized()
        else:
            super().run()
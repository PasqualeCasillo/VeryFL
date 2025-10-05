import logging
logger = logging.getLogger(__name__)

from torch.utils.data import DataLoader

from config.algorithm import Algorithm
from server.aggregation_alg.fedavg import fedavgAggregator
from client.clients import Client, BaseClient, SignClient
from client.trainer.fedproxTrainer import fedproxTrainer
from client.trainer.SignTrainer import SignTrainer
from model.ModelFactory import ModelFactory
from dataset.DatasetFactory import DatasetFactory
from dataset.DatasetSpliter import DatasetSpliter
from chainfl.interact import chain_proxy

# NUOVE IMPORTAZIONI
from utils import MetricsCalculator, MetricsLogger, MetricsPlotter
import torch

class Task:
    def __init__(self, global_args: dict, train_args: dict, algorithm: Algorithm):
        self.global_args = global_args
        self.train_args = train_args
        self.model = None
        
        # NUOVO: Inizializza sistema metriche
        self.metrics_calculator = MetricsCalculator()
        self.metrics_logger = MetricsLogger(save_dir='results')
        self.metrics_plotter = MetricsPlotter(save_dir='results/plots')
        
        logger.info("Constructing dataset %s from dataset Factory", global_args.get('dataset'))
        self.train_dataset = DatasetFactory().get_dataset(global_args.get('dataset'), True)
        self.test_dataset = DatasetFactory().get_dataset(global_args.get('dataset'), False)
        
        logger.info("Constructing Model from model factory with model %s and class_num %d", 
                    global_args['model'], global_args['class_num'])
        self.model = ModelFactory().get_model(model=self.global_args.get('model'),
                                             class_num=self.global_args.get('class_num'))
        
        # logger.info(f"Algorithm: {algorithm}")
        self.server = algorithm.get_server()
        self.server = self.server()
        self.trainer = algorithm.get_trainer()
        self.client = algorithm.get_client()
        
        self.client_list = None
        self.client_pool: list[Client] = []
    
    def __repr__(self) -> str:
        pass
    
    def _construct_dataloader(self):
        logger.info("Constructing dataloader with batch size %d, client_num: %d, non-iid: %s", 
                    self.global_args.get('batch_size'),
                    chain_proxy.get_client_num(), 
                    "True" if self.global_args['non-iid'] else "False")
        batch_size = self.global_args.get('batch_size')
        batch_size = 8 if (batch_size is None) else batch_size
        self.train_dataloader_list = DatasetSpliter().random_split(
            dataset=self.train_dataset,
            client_list=chain_proxy.get_client_list(),
            batch_size=batch_size
        )
        self.test_dataloader = DataLoader(dataset=self.test_dataset, 
                                         batch_size=batch_size, 
                                         shuffle=True)
    
    def _construct_sign(self):
        self.keys_dict = dict()
        self.keys = list()
        sign_num = self.global_args.get('sign_num')
        if (None == sign_num):
            sign_num = 0
            logger.info("No client need to add watermark")
            for ind, (client_id, _) in enumerate(self.client_list.items()):
                self.keys_dict[client_id] = None
        else:
            logger.info(f"{sign_num} client(s) will inject watermark into their models")
            
            for i in range(self.global_args.get('client_num')):
                if i < self.global_args.get('sign_num'):
                    key = chain_proxy.construct_sign(self.global_args)
                    self.keys.append(key)
                else:
                    self.keys.append(None)
            for ind, (client_id, _) in enumerate(self.client_list.items()):
                self.keys_dict[client_id] = self.keys[ind]
            
            tmp_args = chain_proxy.construct_sign(self.global_args)
            self.model = ModelFactory().get_sign_model(
                model=self.global_args.get('model'),
                class_num=self.global_args.get('class_num'),
                in_channels=self.global_args.get('in_channels'),
                watermark_args=tmp_args
            )
        return
    
    def _regist_client(self):
        """Registra client sulla blockchain"""
        # logger.info("=" * 60)
        # logger.info("CLIENT REGISTRATION PHASE")
        # logger.info("=" * 60)

        for i in range(self.global_args['client_num']):
            client_id = chain_proxy.client_regist()

            # Verifica registrazione
            verification = chain_proxy.verify_client_registration(client_id)
            if verification['registered']:
                logger.debug(f"Client {client_id} verified on blockchain (ID: {verification['blockchain_id']})")
            else:
                logger.warning(f"Client {client_id} verification failed: {verification['reason']}")

        self.client_list = chain_proxy.get_client_list()

        # Riepilogo finale
        # logger.info("=" * 60)
        # logger.info("REGISTRATION SUMMARY")
        # logger.info("=" * 60)

        registered = chain_proxy.get_all_registered_clients()
        logger.info(f"Total clients registered on blockchain: {len(registered)}")

        # for client in registered:
        #     logger.info(f"  Client {client['local_id']}: "
        #                f"Blockchain ID={client['blockchain_id']}, "
        #                f"Address={client['address'][:10]}...")

        # logger.info("=" * 60)
    
    def _construct_client(self):
        for client_id, _ in self.client_list.items():
            new_client = self.client(
                client_id, 
                self.train_dataloader_list[client_id], 
                self.model,
                self.trainer, 
                self.train_args, 
                self.test_dataloader, 
                self.keys_dict[client_id]
            )
            self.client_pool.append(new_client)
    
    # NUOVA FUNZIONE: Crea modello per calcolo metriche
    def _create_aggregated_model(self, global_state_dict):
        """Crea un modello con i pesi aggregati per calcolare le metriche"""
        eval_model = ModelFactory().get_model(
            model=self.global_args.get('model'),
            class_num=self.global_args.get('class_num')
        )
        eval_model.load_state_dict(global_state_dict)
        return eval_model
    
    def run(self):
        self._regist_client()
        self._construct_dataloader()
        self._construct_sign()
        self._construct_client()
        
        device = self.train_args.get('device', 'cpu')
        
        # NUOVO: Traccia le metriche durante il training
        for i in range(self.global_args['communication_round']):
            logger.info(f"Starting FL round {i+1}/{self.global_args['communication_round']}")
            
            # Lista per tracciare le loss dei client
            round_losses = []
            
            # Client training phase
            for client in self.client_pool:
                client.train(epoch=i)
                test_result = client.test(epoch=i)
                client.sign_test(epoch=i)
                
                # Traccia la loss del client
                if test_result and 'loss' in test_result:
                    round_losses.append(test_result['loss'])
            
            # NEW: Upload models to IPFS before aggregation
            logger.info("Uploading client models to IPFS...")
            for idx, client in enumerate(self.client_pool):
                upload_params = {
                    'epoch': i,
                    'state_dict': client.get_model_state_dict(),
                    'client_id': client.client_id,
                    'timestamp': None
                }
                result = chain_proxy.upload_model(upload_params)
                logger.info(f"Client {client.client_id} upload result: {result}")
            
            # Server aggregation
            logger.info("Starting model aggregation...")
            self.server.receive_upload(self.client_pool)
            global_model = self.server.aggregate()
            
            # NEW: Upload global model to IPFS
            logger.info("Uploading global model to IPFS...")
            global_upload_params = {
                'epoch': i,
                'state_dict': global_model,
                'client_id': 'global_server',
                'timestamp': None
            }
            global_result = chain_proxy.upload_model(global_upload_params)
            logger.info(f"Global model upload result: {global_result}")
            
            # NUOVO: Calcola metriche sul modello globale
            logger.info("Calculating global metrics...")
            eval_model = self._create_aggregated_model(global_model)
            eval_model.to(device)
            
            # Metriche globali
            global_metrics = self.metrics_calculator.calculate_all_metrics(
                eval_model, 
                self.test_dataloader, 
                device
            )
            
            # Metriche per-nodo
            node_metrics = {}
            for client in self.client_pool:
                client_model = self._create_aggregated_model(client.get_model_state_dict())
                client_model.to(device)
                node_metrics[client.client_id] = self.metrics_calculator.calculate_all_metrics(
                    client_model,
                    self.test_dataloader,
                    device
                )
            
            # Loss media del round
            avg_loss = sum(round_losses) / len(round_losses) if round_losses else 0.0
            
            # Log metriche
            self.metrics_logger.log_round(i + 1, global_metrics, node_metrics, avg_loss)
            
            # Log su console
            logger.info(f"Round {i+1} - Global Metrics: "
                       f"Acc: {global_metrics['accuracy']:.4f}, "
                       f"F1: {global_metrics['f1']:.4f}, "
                       f"Loss: {avg_loss:.4f}")
            
            # Distribute global model to clients
            logger.info("Distributing global model to clients...")
            for client in self.client_pool:
                downloaded_model = chain_proxy.download_model()
                if downloaded_model and 'state_dict' in downloaded_model:
                    client.load_state_dict(downloaded_model['state_dict'])
                    logger.info(f"Client {client.client_id} received model from IPFS")
                else:
                    client.load_state_dict(global_model)
                    logger.info(f"Client {client.client_id} received model via fallback")
            
            # NUOVO: Genera plot periodicamente (ogni 10 round) e alla fine
            if (i + 1) % 10 == 0 or (i + 1) == self.global_args['communication_round']:
                logger.info(f"Generating plots at round {i+1}...")
                self.metrics_plotter.plot_all(self.metrics_logger.metrics)
        
        # NUOVO: Salva metriche finali e genera plot
        logger.info("Training completed. Saving final metrics and plots...")
        self.metrics_logger.save('final_metrics.json')
        self.metrics_plotter.plot_all(self.metrics_logger.metrics)
        logger.info(f"Results saved to: {self.metrics_logger.save_dir}")
        logger.info(f"Plots saved to: {self.metrics_plotter.save_dir}")
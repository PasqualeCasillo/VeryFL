# test_decentralized_crypto.py

import logging
import argparse
from DecentralizedTask import DecentralizedTask
import config.benchmark
from config.log import set_log_config

logger = logging.getLogger(__name__)
set_log_config()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, 
                       default="DecentralizedFashionMNISTCrypto",
                       help="Benchmark with encryption enabled")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("DECENTRALIZED FL WITH ENCRYPTION AND GROUP SIGNATURES")
    logger.info("=" * 60)
    logger.info(f"Benchmark: {args.benchmark}")
    
    benchmark = config.benchmark.get_benchmark(args.benchmark)
    global_args, train_args, algorithm = benchmark.get_args()
    
    logger.info(f"Dataset: {global_args['dataset']}")
    logger.info(f"Model: {global_args['model']}")
    logger.info(f"Nodes: {global_args['client_num']}")
    logger.info(f"Rounds: {global_args['communication_round']}")
    logger.info(f"Encryption: {global_args.get('enable_encryption', False)}")
    logger.info(f"Aggregation: {global_args['aggregation_method']}")
    logger.info("=" * 60)
    
    decentralized_task = DecentralizedTask(
        global_args=global_args, 
        train_args=train_args, 
        algorithm=algorithm
    )
    
    decentralized_task.run()
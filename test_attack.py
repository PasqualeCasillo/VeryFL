# test_attack.py
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
                       default="DecentralizedPowerGridAttack",
                       help="Benchmark with attack simulation")
    args = parser.parse_args()
    
    logger.info(f"Starting decentralized FL with Byzantine attack")
    
    benchmark = config.benchmark.get_benchmark(args.benchmark)
    global_args, train_args, algorithm = benchmark.get_args()
    
    # Log configurazione attacco
    logger.info("=" * 60)
    logger.info("ATTACK CONFIGURATION")
    logger.info(f"Attack Type: {global_args['attack_type']}")
    logger.info(f"Byzantine Ratio: {global_args['byzantine_ratio']}")
    logger.info(f"Attack Start Round: {global_args['attack_start_round']}")
    logger.info(f"Aggregation Method: {global_args['aggregation_method']}")
    logger.info("=" * 60)
    
    decentralized_task = DecentralizedTask(
        global_args=global_args, 
        train_args=train_args, 
        algorithm=algorithm
    )
    
    decentralized_task.run()
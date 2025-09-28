# test_decentralized.py
import logging
import argparse
from DecentralizedTask import DecentralizedTask
import config.benchmark
from config.log import set_log_config

logger = logging.getLogger(__name__)
set_log_config()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, default="DecentralizedFashionMNIST", 
                       help="Running Benchmark (See ./config/benchmark.py)")
    args = parser.parse_args()
    
    logger.info(f"Starting decentralized FL with benchmark {args.benchmark}")
    
    benchmark = config.benchmark.get_benchmark(args.benchmark)
    global_args, train_args, algorithm = benchmark.get_args()
    
    decentralized_task = DecentralizedTask(
        global_args=global_args, 
        train_args=train_args, 
        algorithm=algorithm
    )
    
    decentralized_task.run()
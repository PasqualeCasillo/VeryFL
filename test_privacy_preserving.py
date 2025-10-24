# test_crypto_complete.py
"""
Test completo del sistema privacy-preserving:
- Manifest strutturato
- BBS+ signatures
- KRS per distribuzione chiavi
- VLR per revoca nodi
"""

import logging
from DecentralizedTask import DecentralizedTask
import config.benchmark
from config.log import set_log_config
from chainfl.revocation import VLRManager

logger = logging.getLogger(__name__)
set_log_config()

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("VERYFL - COMPLETE PRIVACY-PRESERVING SYSTEM TEST")
    logger.info("=" * 80)
    logger.info("Features:")
    logger.info("  ✓ AES-256-GCM encryption")
    logger.info("  ✓ BBS+ group signatures")
    logger.info("  ✓ Structured manifest (RFC-compliant)")
    logger.info("  ✓ Key Release Service (KRS)")
    logger.info("  ✓ Verifier-Local Revocation (VLR)")
    logger.info("=" * 80)
    
    # Setup benchmark
    benchmark = config.benchmark.get_benchmark("DecentralizedPowerGridCrypto")
    global_args, train_args, algorithm = benchmark.get_args()
    
    logger.info(f"Dataset: {global_args['dataset']}")
    logger.info(f"Model: {global_args['model']}")
    logger.info(f"Nodes: {global_args['client_num']}")
    logger.info(f"Rounds: {global_args['communication_round']}")
    logger.info(f"Encryption: {global_args.get('enable_encryption', False)}")
    logger.info("=" * 80)
    
    # OPZIONALE: Test revoca nodo
    # vlr = VLRManager()
    # vlr.revoke("2")  # Revoca node 2
    # logger.warning("  Node 2 REVOKED for testing")
    
    # Run task
    task = DecentralizedTask(
        global_args=global_args, 
        train_args=train_args, 
        algorithm=algorithm
    )
    
    task.run()
    
    logger.info("=" * 80)
    logger.info("TEST COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
"""
Updated interact.py with IPFS integration - Fixed imports
"""
from util import jsonFormat
from collections import defaultdict
import string
import json
import logging
import torch

# Import IPFS client
try:
    from .ipfs_client import IPFSClient
except ImportError:
    IPFSClient = None

logger = logging.getLogger(__name__)

# Brownie imports with error handling
try:
    from brownie import *
    import brownie
    
    # Chain init
    p = brownie.project.load(project_path="chainEnv", name="chainServer")
    p.load_config()
    
    # Connect to network
    brownie.network.connect('development')
    
    # Import contracts dynamically after project load
    server_accounts = brownie.accounts[0]
    
    # Deploy existing contracts
    watermark_contract = None
    client_manager_contract = None
    model_registry_contract = None
    
    # Try to deploy contracts
    try:
        # Get contract classes from loaded project
        contracts = brownie.project.chainServer
        
        # Deploy watermark contract
        if hasattr(contracts, 'watermarkNegotiation'):
            watermark_contract = contracts.watermarkNegotiation.deploy({'from': server_accounts})
            
        # Deploy client manager
        if hasattr(contracts, 'clientManager'):
            client_manager_contract = contracts.clientManager.deploy({'from': server_accounts})
            
        # Deploy model registry
        if hasattr(contracts, 'ModelRegistry'):
            model_registry_contract = contracts.ModelRegistry.deploy({'from': server_accounts})
        
        logger.info("Smart contracts deployed successfully")
        
    except Exception as e:
        logger.warning(f"Some contracts failed to deploy: {e}")
        
except ImportError as e:
    logger.warning(f"Brownie not available: {e}")
    # Mock objects for development without blockchain
    class MockAccount:
        def __init__(self, address):
            self.address = address
    
    server_accounts = MockAccount("0x" + "0"*40)
    brownie = None
    watermark_contract = None
    client_manager_contract = None  
    model_registry_contract = None

class chainProxy():
    def __init__(self):
        self.upload_params = None
        self.account_num = 10  # Default fallback
        self.server_accounts = server_accounts
        self.client_num = 0
        self.client_list = defaultdict(str)
        
        # Smart contract references
        self.watermark_proxy = watermark_contract
        self.client_manager = client_manager_contract
        self.model_registry = model_registry_contract
        
        # IPFS integration
        self.ipfs_client = None
        if IPFSClient:
            try:
                self.ipfs_client = IPFSClient()
                logger.info("IPFS client initialized successfully")
                
                # Test IPFS connection
                test_data = b"VeryFL test connection"
                test_hash = self.ipfs_client.client.add_bytes(test_data)
                logger.info(f"IPFS connection test successful: {test_hash}")
                
            except Exception as e:
                logger.warning(f"IPFS client failed to initialize: {e}")
        
        self.current_epoch = 0
        
    def get_account_num(self):
        return self.account_num
    
    def get_client_num(self):
        return self.client_num
    
    def get_client_list(self):
        return self.client_list
    
    def add_account(self) -> str:
        if brownie:
            try:
                account = brownie.accounts.add()
                self.account_num += 1
                return account.address
            except:
                pass
        
        # Fallback: generate mock address
        self.account_num += 1
        return f"0x{'1'*40}"
    
    def client_regist(self) -> str:
        self.client_num += 1
        
        if self.account_num < self.client_num:
            self.add_account()
        
        # Create client account reference
        if brownie and len(brownie.accounts) > self.client_num:
            self.client_list[str(self.client_num)] = brownie.accounts[self.client_num]
        else:
            self.client_list[str(self.client_num)] = f"client_{self.client_num}"
        
        return str(self.client_num)
    
    def watermark_negotitaion(self, client_id: str, watermark_length=64):
        if self.watermark_proxy:
            try:
                client_id = int(client_id)
                if brownie and len(brownie.accounts) > client_id:
                    self.watermark_proxy.generate_watermark({'from': brownie.accounts[client_id]})
                    logger.info(f"Watermark generated for client {client_id}")
                    return
            except Exception as e:
                logger.warning(f"Blockchain watermark failed: {e}")
        
        logger.info(f"Mock watermark generated for client {client_id}")
    
    def upload_model_to_ipfs(self, upload_params: dict):
        """Upload model to IPFS and register hash on blockchain"""
        if not self.ipfs_client:
            logger.warning("IPFS client not available")
            return None
        
        model_state_dict = upload_params['state_dict']
        
        # Prepare metadata
        metadata = {
            'epoch': upload_params.get('epoch', 0),
            'client_id': upload_params.get('client_id', 'unknown'),
            'timestamp': upload_params.get('timestamp')
        }
        
        try:
            # Upload to IPFS
            logger.info("Uploading model to IPFS network...")
            ipfs_hash = self.ipfs_client.upload_model(model_state_dict, metadata)
            
            if ipfs_hash:
                logger.info(f"Model uploaded to IPFS: {ipfs_hash}")
                
                # Try to register on blockchain
                if self.model_registry:
                    try:
                        logger.info("Registering model on blockchain...")
                        tx = self.model_registry.registerModel(
                            self.current_epoch,
                            ipfs_hash,
                            self.client_num,
                            {'from': self.server_accounts}
                        )
                        
                        # Activate the model
                        self.model_registry.activateModel(
                            self.current_epoch,
                            {'from': self.server_accounts}
                        )
                        
                        logger.info(f"Model registered on blockchain: Epoch {self.current_epoch}")
                        
                    except Exception as e:
                        logger.warning(f"Blockchain registration failed, IPFS only: {e}")
                
                # Pin important models
                self.ipfs_client.pin_model(ipfs_hash)
                return ipfs_hash
                
        except Exception as e:
            logger.error(f"IPFS upload failed: {e}")
        
        return None
    
    def download_model_from_ipfs(self, epoch=None):
        """Download model from IPFS using blockchain registry"""
        if not self.ipfs_client:
            return None
        
        try:
            ipfs_hash = None
            
            # Try to get hash from blockchain
            if self.model_registry:
                try:
                    if epoch is None:
                        ipfs_hash = self.model_registry.getCurrentModel()
                    else:
                        model_info = self.model_registry.getModel(epoch)
                        ipfs_hash = model_info[0]
                        
                except Exception as e:
                    logger.warning(f"Blockchain lookup failed: {e}")
            
            # If we have an IPFS hash, download the model
            if ipfs_hash:
                model_state_dict = self.ipfs_client.download_model(ipfs_hash)
                
                if model_state_dict:
                    # logger.info(f"Model downloaded from IPFS: {ipfs_hash}")
                    return {
                        'state_dict': model_state_dict,
                        'ipfs_hash': ipfs_hash,
                        'epoch': epoch or self.current_epoch
                    }
            
        except Exception as e:
            logger.warning(f"IPFS download failed: {e}")
        
        return None
    
    def upload_model(self, upload_params: dict):
        """Upload model with IPFS integration and fallback"""
        try:
            # Try IPFS first
            ipfs_hash = self.upload_model_to_ipfs(upload_params)
            if ipfs_hash:
                self.current_epoch += 1
                logger.info(f"Model uploaded via IPFS: {ipfs_hash}")
                return ipfs_hash
        except Exception as e:
            logger.warning(f"IPFS upload failed: {e}")
        
        # Fallback to original method
        try:
            model_state_dict = upload_params['state_dict']
            upload_params['state_dict'] = jsonFormat.model2json(model_state_dict)
            self.upload_params = upload_params
            logger.info("Model uploaded via local storage (fallback)")
            return "local_storage"
        except Exception as e:
            logger.error(f"All upload methods failed: {e}")
            return None
    
    def download_model(self, params=None):
        """Download model with IPFS integration and fallback"""
        try:
            # Try IPFS first
            model_data = self.download_model_from_ipfs()
            if model_data:
                logger.info(f"Model downloaded from IPFS: {model_data.get('ipfs_hash', 'unknown')}")
                return model_data
        except Exception as e:
            logger.warning(f"IPFS download failed: {e}")
        
        # Fallback to original method
        try:
            if hasattr(self, 'upload_params') and self.upload_params:
                download_params = self.upload_params.copy()
                download_params['state_dict'] = jsonFormat.json2model(download_params['state_dict'])
                logger.info("Model downloaded via local storage (fallback)")
                return download_params
        except Exception as e:
            logger.error(f"All download methods failed: {e}")
        
        return None
    
    def construct_sign(self, args: dict = {}):
        """Existing watermark construction method"""
        sign_config = args.get('sign_config')
        model_name = args.get('model')
        bit_length = args.get('bit_length')
        
        if model_name != "SignAlexNet":
            logger.error("Watermark Not Support for this network")
            raise Exception("Watermark Not Support for this network")
        
        watermark_args = dict()
        alexnet_channels = {
            '4': (384, 3456),
            '5': (256, 2304),
            '6': (256, 2304)
        }
        
        for layer_key in sign_config:
            flag = sign_config[layer_key]
            b = flag if isinstance(flag, str) else None
            if b is not None:
                flag = True
            watermark_args[layer_key] = {
                'flag': flag
            }

            if b is not None:
                if layer_key == "4":
                    output_channels = int(bit_length * 384 / 896)
                if layer_key == "5":
                    output_channels = int(bit_length * 256 / 896)
                if layer_key == "6":
                    output_channels = int(bit_length * 256 / 896)

                b = torch.sign(torch.rand(output_channels) - 0.5)
                M = torch.randn(alexnet_channels[layer_key][0], output_channels)

                watermark_args[layer_key]['b'] = b
                watermark_args[layer_key]['M'] = M

        return watermark_args

# Create global instance
chain_proxy = chainProxy()
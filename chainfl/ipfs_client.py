"""
Simple IPFS client for VeryFL integration
Handles basic model storage and retrieval from IPFS network
"""
import ipfshttpclient
import pickle
import logging
import json
from collections import OrderedDict
import torch

logger = logging.getLogger(__name__)

class IPFSClient:
    def __init__(self, ipfs_api='/ip4/127.0.0.1/tcp/5001'):
        """
        Initialize IPFS client
        :param ipfs_api: IPFS daemon API endpoint
        """
        try:
            self.client = ipfshttpclient.connect(ipfs_api)
            logger.info("Connected to IPFS daemon")
        except Exception as e:
            logger.error(f"Failed to connect to IPFS: {e}")
            self.client = None
    
    def upload_model(self, model_state_dict, metadata=None):
        """
        Upload PyTorch model to IPFS
        :param model_state_dict: OrderedDict from model.state_dict()
        :param metadata: Additional metadata dict
        :return: IPFS hash string
        """
        if self.client is None:
            logger.error("IPFS client not connected")
            return None
        
        try:
            # Prepare model data
            model_data = {
                'state_dict': model_state_dict,
                'metadata': metadata or {}
            }
            
            # Serialize to bytes
            model_bytes = pickle.dumps(model_data)
            
            # Upload to IPFS
            result = self.client.add_bytes(model_bytes)
            ipfs_hash = result
            
            logger.info(f"Model uploaded to IPFS: {ipfs_hash}")
            return ipfs_hash
            
        except Exception as e:
            logger.error(f"Failed to upload model to IPFS: {e}")
            return None
    
    def download_model(self, ipfs_hash):
        """
        Download PyTorch model from IPFS
        :param ipfs_hash: IPFS hash string
        :return: model state_dict or None
        """
        if self.client is None:
            logger.error("IPFS client not connected")
            return None
        
        try:
            # Download from IPFS
            model_bytes = self.client.cat(ipfs_hash)
            
            # Deserialize
            model_data = pickle.loads(model_bytes)
            
            logger.info(f"Model downloaded from IPFS: {ipfs_hash}")
            return model_data['state_dict']
            
        except Exception as e:
            logger.error(f"Failed to download model from IPFS: {e}")
            return None
    
    def pin_model(self, ipfs_hash):
        """
        Pin model to prevent garbage collection
        :param ipfs_hash: IPFS hash to pin
        """
        if self.client is None:
            return False
        
        try:
            self.client.pin.add(ipfs_hash)
            logger.info(f"Model pinned: {ipfs_hash}")
            return True
        except Exception as e:
            logger.error(f"Failed to pin model: {e}")
            return False
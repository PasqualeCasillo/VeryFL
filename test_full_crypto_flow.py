# test_full_crypto_flow.py

import sys
sys.path.insert(0, '.')

import asyncio
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from node.DecentralizedNode import DecentralizedNode
from protocols.AuctionProtocol import AuctionProtocol
from chainfl.auction_proxy import auction_chain_proxy
from chainfl.ipfs_client import IPFSClient
from config.attack_config import AttackConfig

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)

class DummyTrainer:
    def __init__(self, model, dataloader, criterion, args):
        self.model = model
        self.args = args
    
    def train(self, num_steps):
        return [{'loss': 0.5}]

async def test_full_flow():
    print("=" * 60)
    print("TESTING FULL CRYPTO FLOW")
    print("=" * 60)
    
    # 1. Create nodes
    nodes = []
    for i in range(3):
        dummy_data = torch.randn(100, 10)
        dummy_labels = torch.randint(0, 2, (100,))
        dataset = TensorDataset(dummy_data, dummy_labels)
        dataloader = DataLoader(dataset, batch_size=32)
        
        model = DummyModel()
        
        node = DecentralizedNode(
            node_id=str(i+1),
            model=model,
            dataloader=dataloader,
            trainer_class=DummyTrainer,
            train_args={'device': 'cpu', 'num_steps': 1},
            test_dataloader=None,
            num_classes=2
        )
        nodes.append(node)
    
    print(f"Created {len(nodes)} nodes")
    
    # 2. Setup protocol with crypto
    attack_config = AttackConfig(
        attack_type='none',
        byzantine_ratio=0.0,
        attack_start_round=0
    )
    
    protocol = AuctionProtocol(
        blockchain_proxy=auction_chain_proxy,
        timeout_seconds=180,
        aggregation_method='fedavg',
        attack_config=attack_config
    )
    
    # 3. Setup group keys
    group_keys = protocol.setup_group_keys(nodes)
    print(f"Generated {len(group_keys)} BBS+ keypairs")
    
    # 4. Setup IPFS client
    ipfs_client = IPFSClient()
    ipfs_client.set_group_keys(group_keys)
    ipfs_client.enable_encryption(True)
    print("IPFS client configured with encryption")
    
    # 5. Test upload from node 0
    print("\nTesting encrypted upload...")
    node = nodes[0]
    model_state = node.get_model_state_dict()
    metadata = {'node_id': node.node_id, 'round': 1}
    
    cid_manifest, key_data = ipfs_client.upload_model_secured(
        model_state,
        metadata,
        node.bbs_secret_key
    )
    
    if cid_manifest:
        print(f"Upload successful: {cid_manifest}")
        print(f"Encrypted: {key_data.get('encrypted', False)}")
        
        # 6. Test download with verification
        print("\nTesting encrypted download with verification...")
        node.store_encryption_key(cid_manifest, key_data['Kc'])
        
        downloaded = ipfs_client.download_model_secured(
            cid_manifest,
            key_data['Kc']
        )
        
        if downloaded:
            print("Download and verification successful")
            
            # 7. Verify content
            original_keys = set(model_state.keys())
            downloaded_keys = set(downloaded.keys())
            
            if original_keys == downloaded_keys:
                print("Content integrity: PASSED")
                
                # Check one tensor
                orig_tensor = model_state['fc.weight']
                down_tensor = downloaded['fc.weight']
                
                if torch.allclose(orig_tensor, down_tensor):
                    print("Tensor equality: PASSED")
                else:
                    print("Tensor equality: FAILED")
            else:
                print("Content integrity: FAILED")
        else:
            print("Download failed")
    else:
        print("Upload failed")
    
    print("\n" + "=" * 60)
    print("FULL CRYPTO FLOW TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_full_flow())
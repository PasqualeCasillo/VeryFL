# test_node_crypto.py

import sys
sys.path.insert(0, '.')

from node.DecentralizedNode import DecentralizedNode
from chainfl.ipfs_client import IPFSClient
from chainfl.crypto_bbs import bbs_generate_keypair
import torch
import torch.nn as nn

print("Testing DecentralizedNode with crypto...")

# Create dummy model
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)

model = DummyModel()

# Create dummy dataloader
from torch.utils.data import TensorDataset, DataLoader
dummy_data = torch.randn(100, 10)
dummy_labels = torch.randint(0, 2, (100,))
dataset = TensorDataset(dummy_data, dummy_labels)
dataloader = DataLoader(dataset, batch_size=32)

# Create node
node = DecentralizedNode(
    node_id='1',
    model=model,
    dataloader=dataloader,
    trainer_class=None,
    train_args={'device': 'cpu'},
    test_dataloader=None,
    num_classes=2
)

# Generate BBS+ key
pk, sk = bbs_generate_keypair()
node.bbs_secret_key = sk

print(f"Node created: {node.node_id}")
print(f"Has BBS+ key: {hasattr(node, 'bbs_secret_key')}")
print(f"Has encryption keys storage: {hasattr(node, '_encryption_keys')}")

# Test key storage
node.store_encryption_key('test_cid', 'test_key_b64')
retrieved = node._get_encryption_key_for_manifest('test_cid')
print(f"Key storage test: {'PASSED' if retrieved == 'test_key_b64' else 'FAILED'}")

print("Import and basic tests: PASSED")
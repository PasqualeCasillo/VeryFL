# dataset/DatasetSpliter.py

import logging
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from collections import defaultdict

logger = logging.getLogger(__name__)

class DatasetSpliter:
    def __init__(self) -> None:
        return
    
    def random_split(self, dataset: Dataset, client_list: dict, batch_size: int = 32) -> dict:
        """Split dataset randomly with NO overlap"""
        client_num = len(client_list)
        total_samples = len(dataset)
        samples_per_client = total_samples // client_num
        
        all_indices = list(range(total_samples))
        random.shuffle(all_indices)
        
        dataloaders = {}
        for idx, (client_id, _) in enumerate(client_list.items()):
            start = idx * samples_per_client
            end = (idx + 1) * samples_per_client if idx < client_num - 1 else total_samples
            client_indices = all_indices[start:end]
            
            sampler = SubsetRandomSampler(client_indices)
            dataloaders[client_id] = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=4
            )
            logger.info(f"Client {client_id}: {len(client_indices)} samples")
        
        return dataloaders
    
    def _sample_dirichlet(self, dataset: Dataset, client_list: dict, alpha: int) -> defaultdict:
        client_num = len(client_list.keys())
        per_class_list = defaultdict(list)
        
        for ind, (_, label) in enumerate(dataset):
           per_class_list[label].append(ind)
        
        class_num = len(per_class_list.keys())
        per_client_list = defaultdict(list)
        
        for n in range(class_num):
            random.shuffle(per_class_list[n])
            class_size = len(per_class_list[n])
            sampled_probabilities = class_size * np.random.dirichlet(np.array(client_num * [alpha]))
            
            for ind, (client_id, _) in enumerate(client_list.items()):
                no_imgs = int(round(sampled_probabilities[ind]))
                sampled_list = per_class_list[n][:min(len(per_class_list[n]), no_imgs)]
                per_client_list[client_id].extend(sampled_list)
                per_class_list[n] = per_class_list[n][min(len(per_class_list[n]), no_imgs):]
        
        return per_client_list 
        
    def dirichlet_split(self, dataset: Dataset, client_list: dict, batch_size: int = 32, alpha: int = 1) -> dict:
        split_list = self._sample_dirichlet(dataset, client_list, alpha)
        dataloaders = defaultdict(DataLoader)
        
        for client_id, _ in client_list.items():
            indices = split_list[client_id]
            dataloaders[client_id] = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                sampler=SubsetRandomSampler(indices),
                num_workers=4
            )
            logger.info(f"Client {client_id}: {len(indices)} samples")
        
        return dataloaders
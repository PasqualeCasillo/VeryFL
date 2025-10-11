# utils/attack_utils.py
import torch
import logging
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class LabelFlippingDataset(Dataset):
    """Wrapper dataset che flippa le label"""
    def __init__(self, original_dataset, flip_probability=1.0, num_classes=10):
        self.dataset = original_dataset
        self.flip_probability = flip_probability
        self.num_classes = num_classes
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        
        # Converti label a int
        label = int(label)
        
        # Verifica che label sia valida
        if label >= self.num_classes or label < 0:
            # Fallback: usa label 0 se invalida
            label = 0
        
        # Flippa la label con probabilità flip_probability
        if torch.rand(1).item() < self.flip_probability:
            # Scegli casualmente un'altra classe (esclusa quella corretta)
            available_labels = list(range(self.num_classes))
            if label in available_labels:
                available_labels.remove(label)
            
            if available_labels:
                flipped_label = available_labels[torch.randint(0, len(available_labels), (1,)).item()]
                return data, flipped_label
        
        return data, label
    
def _validate_num_classes(self):
        """Valida che num_classes sia corretto"""
        sample_labels = []
        for i in range(min(100, len(self.dataset))):
            _, label = self.dataset[i]
            sample_labels.append(int(label))
        
        max_label = max(sample_labels)
        if max_label >= self.num_classes:
            raise ValueError(f"num_classes={self.num_classes} ma trovata label {max_label}")


def create_flipped_dataloader(dataloader, flip_probability=1.0, num_classes=10):
    """
    Crea un nuovo dataloader con label flippate
    """
    from torch.utils.data import DataLoader
    
    original_dataset = dataloader.dataset
    
    # FIX: Determina num_classes dal dataset se non specificato
    if num_classes is None:
        # Scansiona dataset per trovare max label
        max_label = 0
        for i in range(min(1000, len(original_dataset))):
            _, label = original_dataset[i]
            max_label = max(max_label, int(label))
        num_classes = max_label + 1
    
    flipped_dataset = LabelFlippingDataset(original_dataset, flip_probability, num_classes)
    
    flipped_dataloader = DataLoader(
        dataset=flipped_dataset,
        batch_size=dataloader.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    logger.debug(f"Created flipped dataloader: flip_prob={flip_probability}, num_classes={num_classes}")
    return flipped_dataloader
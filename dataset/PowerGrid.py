# dataset/PowerGrid.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from config.dataset import dataset_file_path
import os
import logging

logger = logging.getLogger(__name__)

class PowerGridDataset(Dataset):
    """
    Power Grid dataset per classificazione binaria (Natural vs Attack).
    Dataset per rilevamento anomalie in smart grid elettriche.
    
    Features: 128 misure elettriche (tensione, corrente, frequenza, logs)
    Target: 'Natural' (0) vs 'Attack' (1)
    """
    
    def __init__(self, root, train=True, transform=None, train_split=0.8):
        self.root = root
        self.train = train
        self.transform = transform
        
        # Percorso CSV (modifica secondo la tua struttura)
        csv_path = os.path.join(root, 'archive/binaryAllNaturalPlusNormalVsAttacks/data1.csv')
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Dataset non trovato in {csv_path}\n"
                f"Scarica da: https://www.kaggle.com/datasets/...\n"
                f"E posiziona data1.csv in {root}"
            )
        
        logger.debug(f"Caricamento PowerGrid dataset da {csv_path}")
        
        # Carica CSV
        df = pd.read_csv(csv_path)
        logger.debug(f"Dataset shape: {df.shape}")         
        # Separa features e target
        X = df.drop('marker', axis=1)
        y = df['marker']
        
        # Preprocessing robusto
        X = self._preprocess_features(X)
        y_encoded = self._encode_labels(y)
        
        # Normalizzazione
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Controllo finale NaN/Inf
        X_scaled = self._handle_edge_cases(X_scaled)
        
        # Split train/test deterministico
        train_data, train_labels, test_data, test_labels = self._split_data(
            X_scaled, y_encoded, train_split
        )
        
        if self.train:
            self.data = torch.FloatTensor(train_data)
            self.targets = torch.LongTensor(train_labels)
            logger.info(f"Train set: {len(self.data)} samples")
        else:
            self.data = torch.FloatTensor(test_data)
            self.targets = torch.LongTensor(test_labels)
            logger.info(f"Test set: {len(self.data)} samples")
        
        # Log distribuzione classi
        unique, counts = np.unique(self.targets.numpy(), return_counts=True)
        for label, count in zip(unique, counts):
            logger.info(f"  Classe {label}: {count} samples ({count/len(self.targets)*100:.1f}%)")
    
    def _preprocess_features(self, X):
        """Preprocessing robusto delle features"""
        # Solo colonne numeriche
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_columns]
        
        # Sostituisci infiniti con NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Riempi NaN con mediana per colonna
        X = X.fillna(X.median())
        
        # Clipping valori estremi
        X = X.clip(-1e10, 1e10)
        
        logger.debug(f"Preprocessing {X.shape[1]} features")
        return X
    
    def _encode_labels(self, y):
        """Encoding labels con check distribuzione"""
        y_mapped = y.map({'Natural': 0, 'Attack': 1})
        
        # Verifica encoding valido
        if y_mapped.isna().any():
            logger.warning(f"Trovati {y_mapped.isna().sum()} label non mappati")
            y_mapped = y_mapped.fillna(0)  # Default a Natural
        
        y_encoded = y_mapped.values
        
        unique, counts = np.unique(y_encoded, return_counts=True)
        
        return y_encoded
    
    def _handle_edge_cases(self, X_scaled):
        """Gestione NaN/Inf residui dopo scaling"""
        if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
            logger.warning("Trovati NaN/Inf dopo scaling, applicando nan_to_num")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
        return X_scaled
    
    def _split_data(self, X, y, train_split):
        """Split deterministico train/test"""
        n_samples = len(X)
        n_train = int(n_samples * train_split)
        
        # Shuffle con seed fisso per riproducibilità
        np.random.seed(42)
        indices = np.random.permutation(n_samples)
        
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        return (
            X[train_indices], y[train_indices],
            X[test_indices], y[test_indices]
        )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, target


def get_powergrid(train=True):
    """
    Factory function per ottenere il dataset PowerGrid.
    
    Args:
        train (bool): True per train set, False per test set
        
    Returns:
        PowerGridDataset: Dataset instance
    """
    return PowerGridDataset(
        root=dataset_file_path,
        train=train
    )
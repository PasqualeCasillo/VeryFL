import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from config.dataset import dataset_file_path
import os

class PowerGridDataset(Dataset):
    """
    Power Grid dataset per classificazione binaria (Natural vs Attack).
    Gestione robusta di NaN, infiniti e normalizzazione.
    """
    
    def __init__(self, root, train=True, transform=None, train_split=0.8):
        self.root = root
        self.train = train
        self.transform = transform
        
        # Carica il CSV
        csv_path = os.path.join(root, 'archive/binaryAllNaturalPlusNormalVsAttacks/data1.csv')
        df = pd.read_csv(csv_path)
        
        print(f"PowerGrid dataset: {df.shape[0]} righe, {df.shape[1]} colonne")
        
        # Separa features e target
        X = df.drop('marker', axis=1)
        y = df['marker']
        
        # Rimuovi colonne non numeriche
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_columns]
        
        # Sostituisci infiniti con NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Gestisci NaN con mediana
        X = X.fillna(X.median())
        
        # Clipping per valori estremamente grandi
        if np.any(np.abs(X.values) > 1e10):
            X = X.clip(-1e10, 1e10)
        
        y_mapped = y.map({'Natural': 0, 'Attack': 1}).values

        # PERMUTATION TEST - Decommenta questa riga per testare con label casuali
        # y_encoded = np.random.permutation(y_mapped)  # IL TEST SANITY CHECK FUNZIONA. con label casuali rimane bloccato al baseline di classe maggioritaria (~78%). La differenza di ~7-8% dimostra che il modello impara pattern reali dai dati.
        y_encoded = y_mapped  # Normale
        unique, counts = np.unique(y_encoded, return_counts=True)
        print(f"Distribuzione classi: {dict(zip(unique, counts))}")
        print(f"Percentuale: Natural={counts[0]/len(y_encoded)*100:.1f}%, Attack={counts[1]/len(y_encoded)*100:.1f}%")
        
        # Normalizzazione con StandardScaler (come in Gossipy)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Controllo finale per NaN/Inf
        if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Split train/test con shuffle
        n_samples = len(X_scaled)
        n_train = int(n_samples * train_split)
        
        np.random.seed(42)
        indices = np.random.permutation(n_samples)
        
        if self.train:
            train_indices = indices[:n_train]
            self.data = torch.FloatTensor(X_scaled[train_indices])
            self.targets = torch.LongTensor(y_encoded[train_indices])
            print(f"Train set: {len(self.data)} samples")
        else:
            test_indices = indices[n_train:]
            self.data = torch.FloatTensor(X_scaled[test_indices])
            self.targets = torch.LongTensor(y_encoded[test_indices])
            print(f"Test set: {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, target


def get_powergrid(train=True):
    """Factory function per ottenere il dataset PowerGrid"""
    return PowerGridDataset(
        root=dataset_file_path,
        train=train
    )
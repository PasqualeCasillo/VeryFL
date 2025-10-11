# utils/metrics.py
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

class MetricsCalculator:
    @staticmethod
    def calculate_all_metrics(model, dataloader, device='cpu'):
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0  # ← NUOVO
        num_samples = 0   # ← NUOVO
        
        with torch.no_grad():
            for data, labels in dataloader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                
                # ← NUOVO: Calcola loss
                loss = torch.nn.functional.cross_entropy(outputs, labels, reduction='sum')
                total_loss += loss.item()
                num_samples += labels.size(0)
                
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        accuracy = (all_preds == all_labels).mean()
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        # AUC calculation
        try:
            n_classes = all_probs.shape[1]
            if n_classes == 2:
                auc = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
        except Exception as e:
            print(f"Warning: AUC calculation failed: {e}")
            auc = 0.0
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc),
            'loss': float(total_loss / num_samples)  # ← NUOVO
        }
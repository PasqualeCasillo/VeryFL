# server/aggregation_alg/krum.py
from ..base.baseAggregator import ServerAggregator
import torch
import numpy as np
import logging
import traceback


logger = logging.getLogger(__name__)

class krumAggregator(ServerAggregator):
    def __init__(self, byzantine_ratio=0.25):
        super().__init__()
        self.byzantine_ratio = byzantine_ratio
        
    def _on_before_aggregation(self):
        pass
        
    def _on_after_aggregation(self):
        pass
        
    def test(self):
        pass
    
    def _flatten_model(self, model_dict):
        """
        Converte state_dict in un singolo vettore flat mantenendo l'ordine.
        
        Args:
            model_dict: OrderedDict dal model.state_dict()
            
        Returns:
            torch.Tensor: Vettore flat di tutti i parametri
        """
        vectors = []
        for key in sorted(model_dict.keys()):  # Sort per consistenza
            param = model_dict[key]
            if isinstance(param, torch.Tensor):
                vectors.append(param.flatten())
            else:
                # Fallback per numpy o altri tipi
                vectors.append(torch.tensor(param).flatten())
        
        return torch.cat(vectors)
    
    def _reconstruct_model(self, flat_vector, reference_model):
        """
        Ricostruisce state_dict da vettore flat usando reference model come template.
        
        Args:
            flat_vector: torch.Tensor flat
            reference_model: dict con struttura originale
            
        Returns:
            dict: state_dict ricostruito
        """
        reconstructed = {}
        offset = 0
        
        for key in sorted(reference_model.keys()):
            param_shape = reference_model[key].shape
            param_size = np.prod(param_shape)
            
            # Estrai chunk e reshape
            param_flat = flat_vector[offset:offset + param_size]
            reconstructed[key] = param_flat.reshape(param_shape)
            
            offset += param_size
        
        return reconstructed
        
    def _aggregate_alg(self, raw_client_model_or_grad_list=None):
        if raw_client_model_or_grad_list is None:
            raw_client_model_or_grad_list = self.model_pool

        n = len(raw_client_model_or_grad_list)
        f = int(n * self.byzantine_ratio)

        logger.info(f"=" * 60)
        logger.info(f"KRUM AGGREGATION DEBUG")
        logger.info(f"=" * 60)
        logger.info(f"Total models: n={n}")
        logger.info(f"Byzantine ratio: {self.byzantine_ratio}")
        logger.info(f"Assumed Byzantine: f={f}")
        logger.info(f"Krum requirement: n >= 2f+3 → {n} >= {2*f+3}")

        # Check constraint
        if n < 2 * f + 3:
            logger.error(f"✗ KRUM CONSTRAINT VIOLATED! Using fallback FedAvg")
            return self._fallback_fedavg(raw_client_model_or_grad_list)

        logger.info(f" Krum constraint satisfied")

        # Step 1: Flatten models
        logger.info(f"Flattening {n} models...")
        flat_models = []

        try:
            for i, model in enumerate(raw_client_model_or_grad_list):
                flat = self._flatten_model(model)
                flat_models.append(flat)

                # CALCOLA NORM DI OGNI MODELLO
                norm = torch.norm(flat, p=2)
                logger.info(f"  Model {i}: {len(flat)} params, norm={norm:.2e}")

        except Exception as e:
            logger.error(f"✗ Flatten failed: {e}")
            logger.error(traceback.format_exc())
            return self._fallback_fedavg(raw_client_model_or_grad_list)

        # Step 2: Calcola distanze
        logger.info(f"Computing distance matrix...")
        distances = torch.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.norm(flat_models[i] - flat_models[j], p=2)
                distances[i, j] = dist
                distances[j, i] = dist

        #  LOG MATRICE DISTANZE
        logger.info(f"Distance matrix:")
        for i in range(n):
            row_str = "  " + " ".join(f"{distances[i,j]:.2e}" for j in range(n))
            logger.info(row_str)

        # Step 3: Calcola score
        logger.info(f"Computing Krum scores...")
        scores = []
        n_closest = n - f - 2

        logger.info(f"Using {n_closest} closest neighbors for score")

        for i in range(n):
            dists_sorted, indices_sorted = torch.sort(distances[i])
            score = torch.sum(dists_sorted[1:n_closest + 1])
            scores.append(score.item())

            #  LOG DETTAGLIATO PER OGNI MODELLO
            closest_indices = indices_sorted[1:n_closest + 1].tolist()
            logger.info(f"  Model {i}: score={score:.2e}, closest to: {closest_indices}")

        # Step 4: Selezione
        selected_idx = int(np.argmin(scores))
        selected_score = scores[selected_idx]

        logger.info(f"=" * 60)
        logger.info(f"KRUM SELECTION RESULT")
        logger.info(f"=" * 60)

        for i, score in enumerate(scores):
            marker = "<<< SELECTED" if i == selected_idx else ""
            logger.info(f"  Model {i}: score={score:.2e} {marker}")

        logger.info(f"=" * 60)

        # IDENTIFICA OUTLIERS
        score_mean = np.mean(scores)
        score_std = np.std(scores)
        outliers = []

        for i, s in enumerate(scores):
            if s > score_mean + 2 * score_std:
                outliers.append(i)

        if outliers:
            logger.warning(f"⚠ Detected outliers (likely Byzantine): {outliers}")
        else:
            logger.info(f"No significant outliers detected")

        return raw_client_model_or_grad_list[selected_idx]
    
    def _fallback_fedavg(self, model_list):
        """Fallback semplice a FedAvg se Krum non applicabile"""
        logger.info("Usando FedAvg come fallback")
        
        aggregated = {}
        num_models = len(model_list)
        
        for key in model_list[0].keys():
            aggregated[key] = sum(model[key] for model in model_list) / num_models
        
        return aggregated
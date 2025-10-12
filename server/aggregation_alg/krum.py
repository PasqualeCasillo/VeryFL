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
        logger.info(f"KRUM AGGREGATION")
        logger.info(f"=" * 60)
        logger.info(f"Total models: n={n}")
        logger.info(f"Byzantine ratio: {self.byzantine_ratio}")
        logger.info(f"Assumed Byzantine: f={f}")
        logger.info(f"Krum requirement: n >= 2f+3 → {n} >= {2*f+3}")
    
        #  NUOVO: Check HARD constraint con abort
        if n < 2 * f + 3:
            logger.error(f"=" * 60)
            logger.error(f" KRUM CONSTRAINT VIOLATED!")
            logger.error(f"   Required: n >= {2*f+3}, Got: n={n}")
            logger.error(f"   Cannot guarantee Byzantine-fault-tolerance")
            logger.error(f"=" * 60)
            raise ValueError(
                f"Krum constraint violated: need {2*f+3} models, got {n}. "
                f"System is VULNERABLE with current configuration."
            )
    
        logger.info(f" Krum constraint satisfied")
    
        # Step 1: Flatten models
        logger.info(f"Flattening {n} models...")
        flat_models = []
    
        try:
            for i, model in enumerate(raw_client_model_or_grad_list):
                flat = self._flatten_model(model)
                flat_models.append(flat)
                norm = torch.norm(flat, p=2)
                logger.debug(f"  Model {i}: {len(flat)} params, norm={norm:.2e}")
    
        except Exception as e:
            logger.error(f"✗ Flatten failed: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Model flattening failed: {e}")
    
        # Step 2: Compute pairwise distances
        logger.info(f"Computing distance matrix...")
        distances = torch.zeros((n, n))
    
        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.norm(flat_models[i] - flat_models[j], p=2)
                distances[i, j] = dist
                distances[j, i] = dist
    
        #  NUOVO: Log matrice distanze in formato leggibile
        logger.debug(f"Distance matrix:")
        for i in range(n):
            row_str = "  " + " ".join(f"{distances[i,j]:8.2e}" for j in range(n))
            logger.debug(row_str)
    
        # Step 3: Compute Krum scores
        logger.info(f"Computing Krum scores...")
        n_closest = n - f - 2
    
        if n_closest <= 0:
            raise ValueError(
                f"Invalid Krum configuration: n_closest={n_closest} <= 0. "
                f"Increase number of nodes or decrease byzantine_ratio."
            )
    
        logger.info(f"Using {n_closest} closest neighbors for scoring")
    
        scores = []
        closest_neighbors = []  #  NUOVO: Track neighbors per analysis
    
        for i in range(n):
            dists_sorted, indices_sorted = torch.sort(distances[i])
            
            # Escludi distanza da se stesso (indice 0)
            score = torch.sum(dists_sorted[1:n_closest + 1])
            scores.append(score.item())
            
            closest_indices = indices_sorted[1:n_closest + 1].tolist()
            closest_neighbors.append(closest_indices)
            
            logger.debug(
                f"  Model {i}: score={score:.2e}, "
                f"closest to: {closest_indices}"
            )
    
        # Step 4: Selection + Outlier Detection
        selected_idx = int(np.argmin(scores))
        selected_score = scores[selected_idx]
    
        logger.info(f"=" * 60)
        logger.info(f"KRUM SELECTION RESULT")
        logger.info(f"=" * 60)
    
        #  NUOVO: Statistical outlier detection
        score_array = np.array(scores)
        score_mean = np.mean(score_array)
        score_std = np.std(score_array)
        score_median = np.median(score_array)
        
        logger.info(f"Score statistics:")
        logger.info(f"  Mean: {score_mean:.2e}")
        logger.info(f"  Median: {score_median:.2e}")
        logger.info(f"  Std Dev: {score_std:.2e}")
        logger.info(f"  Min: {np.min(score_array):.2e}")
        logger.info(f"  Max: {np.max(score_array):.2e}")
        logger.info(f"")
    
        # Identifica outliers (score > mean + 2*std)
        outlier_threshold = score_mean + 2 * score_std
        outliers = []
        potential_byzantine = []
    
        for i, s in enumerate(scores):
            is_outlier = s > outlier_threshold
            is_selected = i == selected_idx
            
            marker = ""
            if is_selected:
                marker = " <<< SELECTED"
            elif is_outlier:
                marker = "  OUTLIER (likely Byzantine)"
                outliers.append(i)
                potential_byzantine.append(i)
            
            logger.info(f"  Model {i}: score={s:.2e} {marker}")
    
        logger.info(f"")
        logger.info(f"Summary:")
        logger.info(f"  Selected model: {selected_idx} (score={selected_score:.2e})")
        logger.info(f"  Detected outliers: {len(outliers)} models {outliers}")
        logger.info(f"  Expected Byzantine: ~{f} models")
        
        if len(outliers) > f:
            logger.warning(
                f"  More outliers ({len(outliers)}) than expected Byzantine ({f})!"
            )
        
        logger.info(f"=" * 60)
    
        #  NUOVO: Valida che il modello selezionato NON sia outlier
        if selected_idx in outliers:
            logger.error(
                f" CRITICAL: Selected model {selected_idx} is an outlier! "
                f"This should NOT happen with correct Krum."
            )
            # In teoria questo non dovrebbe mai accadere
            # Se succede, c'è un bug nella logica di Krum
    
        return raw_client_model_or_grad_list[selected_idx]
    
    # def _fallback_fedavg(self, model_list):
    #     """Fallback semplice a FedAvg se Krum non applicabile"""
    #     logger.info("Usando FedAvg come fallback")
        
    #     aggregated = {}
    #     num_models = len(model_list)
        
    #     for key in model_list[0].keys():
    #         aggregated[key] = sum(model[key] for model in model_list) / num_models
        
    #     return aggregated
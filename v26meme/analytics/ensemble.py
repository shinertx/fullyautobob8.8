import numpy as np
import pandas as pd
from loguru import logger
from typing import List, Dict

from v26meme.core.dsl import Alpha
from v26meme.core.state import StateManager

class EnsembleManager:
    """
    Creates and manages a meta-alpha by ensembling signals from top-performing core alphas.
    """
    def __init__(self, state: StateManager, cfg: dict, lakehouse=None):
        self.state = state
        self.cfg = cfg.get('ensemble', {})
        self.portfolio_cfg = cfg.get('portfolio', {})
        self.lakehouse = lakehouse
        if not self.cfg.get('enabled', False):
            logger.info("EnsembleManager is disabled in config.")

    def run(self, active_alphas: List[Alpha], equity: float):
        """
        Main entry point to run the ensemble process.
        """
        if not self.cfg.get('enabled', False):
            return None

        # 1. Select candidate alphas for the ensemble
        candidates = self._select_candidates(active_alphas)
        if not candidates:
            logger.info("Not enough qualified core alphas to build an ensemble.")
            return None

        # 2. Calculate ensemble weights
        ensemble_weights = self._calculate_weights(candidates)
        if not ensemble_weights:
            logger.warning("Failed to calculate ensemble weights.")
            return None
            
        # 3. Generate target portfolio based on the ensemble
        # For now, we will store the ensemble definition.
        # The execution handler will need to be taught how to interpret this.
        ensemble_definition = {
            'id': 'ensemble_meta_alpha',
            'component_alphas': list(ensemble_weights.keys()),
            'weights': ensemble_weights,
            'generated_at': pd.Timestamp.utcnow().isoformat()
        }
        
        self.state.set_ensemble_definition(ensemble_definition)
        logger.success(f"Successfully generated and saved ensemble with {len(candidates)} alphas.")
        
        return ensemble_definition

    def _select_candidates(self, active_alphas: List[Alpha]) -> List[Alpha]:
        """
        Selects the best alphas from the 'core' lane to be part of the ensemble.
        """
        min_alphas = self.cfg.get('min_alphas_for_ensemble', 3)
        max_alphas = self.cfg.get('max_alphas_in_ensemble', 10)

        core_alphas = [alpha for alpha in active_alphas if alpha.lane == 'core']
        
        if len(core_alphas) < min_alphas:
            return []

        # Sort by a robust performance metric, e.g., Sortino ratio
        core_alphas.sort(key=lambda a: a.sortino(), reverse=True)

        return core_alphas[:max_alphas]

    def _calculate_weights(self, candidates: List[Alpha]) -> Dict[str, float]:
        """
        Calculates the weight for each alpha in the ensemble.
        """
        weighting_scheme = self.cfg.get('weighting_scheme', 'inverse_variance')
        
        if weighting_scheme == 'inverse_variance':
            returns_data = {a.id: a.performance.get('all', {}).get('returns', []) for a in candidates}
            max_len = max(len(v) for v in returns_data.values()) if returns_data else 0
            for k, v in returns_data.items():
                v.extend([0.0] * (max_len - len(v)))
            
            df = pd.DataFrame(returns_data)
            if df.empty or df.shape[1] < 1:
                return {}

            inv_var = 1 / df.var().replace(0, np.nan)
            inv_var = inv_var.fillna(inv_var.max()) # Handle NaNs if variance is zero
            total_inv_var = inv_var.sum()
            
            if total_inv_var == 0:
                return {alpha.id: 1.0 / len(candidates) for alpha in candidates} # Fallback to equal weight

            return (inv_var / total_inv_var).to_dict()

        elif weighting_scheme == 'sharpe_ratio':
            weights = {alpha.id: alpha.sharpe() for alpha in candidates}
            # Normalize positive Sharpe ratios
            pos_sharpe_sum = sum(w for w in weights.values() if w > 0)
            if pos_sharpe_sum > 0:
                return {k: max(0, v) / pos_sharpe_sum for k, v in weights.items()}
            else:
                # Fallback to equal weight if no positive Sharpe ratios
                return {alpha.id: 1.0 / len(candidates) for alpha in candidates}

        else: # Default to equal weight
            return {alpha.id: 1.0 / len(candidates) for alpha in candidates}
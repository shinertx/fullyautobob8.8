from typing import List
import numpy as np, pandas as pd
from v26meme.core.dsl import Alpha

class PortfolioOptimizer:
    def __init__(self, portfolio_cfg: dict):
        self.cfg = portfolio_cfg

    def _inv_var_weights(self, returns_df: pd.DataFrame) -> dict:
        if returns_df.shape[1] == 1:
            return {returns_df.columns[0]: 1.0}
        inv_var = 1 / returns_df.var().replace(0, np.nan)
        inv_var = inv_var.fillna(inv_var.max())
        return (inv_var / inv_var.sum()).to_dict()

    def get_weights(self, active_alphas: List[Alpha], regime: str) -> dict:
        """Compute portfolio weights with inverse-variance base, then apply floor & cap.

        Algorithm:
          1. Separate alphas by lane ('core', 'moonshot').
          2. For each lane, build inverse-variance raw weights.
          3. Apply the lane-specific Kelly fraction.
          4. Combine weights from all lanes.
          5. Floor: zero weights below min_allocation_weight.
          6. If zero or one surviving alpha: if one, cap at max_alpha_concentration; return.
          7. Normalize surviving weights to sum=1.
          8. Iteratively cap any weight > max_alpha_concentration, redistribute overflow.
          9. Final normalization.
        """
        if not active_alphas:
            return {}

        # kelly fractions are nested under portfolio.lanes in provided config
        lanes_cfg = (self.cfg.get('portfolio', {}) or {}).get('lanes', {})
        core_kelly = lanes_cfg.get('core', {}).get('kelly_fraction', 0.5)
        moonshot_kelly = lanes_cfg.get('moonshot', {}).get('kelly_fraction', 0.5)

        alphas_by_lane = {'core': [], 'moonshot': []}
        for alpha in active_alphas:
            if alpha.lane in alphas_by_lane:
                alphas_by_lane[alpha.lane].append(alpha)

        combined_weights = {}

        for lane, alphas in alphas_by_lane.items():
            usable = [a for a in alphas if regime in a.performance and a.performance.get(regime, {}).get('n_trades', 0) > 5]
            if not usable:
                temp_regime = 'all'
                usable = [a for a in alphas if 'all' in a.performance and a.performance.get('all', {}).get('n_trades', 0) > 5]
            if not usable:
                continue

            returns_data = {a.id: a.performance.get(regime if usable else temp_regime, {}).get('returns', []) for a in usable}
            max_len = max(len(v) for v in returns_data.values()) if returns_data else 0
            for k, v in returns_data.items():
                v.extend([0.0] * (max_len - len(v)))
            
            df = pd.DataFrame(returns_data)
            if df.empty:
                continue

            raw_lane_weights = self._inv_var_weights(df)
            
            kelly_fraction = core_kelly if lane == 'core' else moonshot_kelly
            
            # Apply Kelly fraction to the lane's weights
            for alpha_id, weight in raw_lane_weights.items():
                combined_weights[alpha_id] = weight * kelly_fraction

        if not combined_weights:
            return {}

        port = self.cfg.get('portfolio', {}) or {}
        floor = float(port.get('min_allocation_weight', 0.0))
        cap = float(port.get('max_alpha_concentration', 1.0))
        
        # Floor pass
        floored = {k: (w if w >= floor else 0.0) for k, w in combined_weights.items()}
        non_zero = {k: w for k, w in floored.items() if w > 0.0}
        
        if not non_zero:
            return {}
        if len(non_zero) == 1:
            k = next(iter(non_zero.keys()))
            return {k: min(non_zero[k], cap, 1.0)}

        # Normalize initial
        s = sum(non_zero.values())
        weights = {k: w / s for k, w in non_zero.items()} if s > 0 else {}
        
        # Iterative cap & redistribute
        for _ in range(len(weights)):  # bounded iterations
            over = {k: w for k, w in weights.items() if w > cap + 1e-12}
            if not over:
                break
            overflow = sum(w - cap for w in over.values())
            for k in over:
                weights[k] = cap
            
            receivers = {k: w for k, w in weights.items() if w < cap - 1e-12}
            if not receivers or overflow <= 0:
                break
            recv_sum = sum(receivers.values())
            if recv_sum <= 0:
                break
            for k, w in receivers.items():
                add = overflow * (w / recv_sum)
                weights[k] = w + add
                
        # Final normalization
        total = sum(weights.values())
        if total > 0 and total > 1.0:
            scale = 1.0 / total
            weights = {k: min(cap, w * scale) for k, w in weights.items()}
            
        return weights

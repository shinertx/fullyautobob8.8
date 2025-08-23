import numpy as np, pandas as pd

class PortfolioOptimizer:
    def __init__(self, cfg: dict):
        self.cfg = cfg['portfolio']

    def _inv_var_weights(self, returns_df: pd.DataFrame) -> dict:
        if returns_df.shape[1] == 1:
            return {returns_df.columns[0]: 1.0}
        inv_var = 1 / returns_df.var().replace(0, np.nan)
        inv_var = inv_var.fillna(inv_var.max())
        return (inv_var / inv_var.sum()).to_dict()

    def get_weights(self, active_alphas: list, regime: str) -> dict:
        """Compute portfolio weights with inverse-variance base, then apply floor & cap.

        Algorithm:
          1. Build inverse-variance raw weights (panel returns aligned, PIT safe — uses only realized returns stored in state).
          2. Drop alphas with insufficient trades (n_trades <=5) for the requested regime, fallback to 'all'.
          3. Floor: zero weights below min_allocation_weight.
          4. If zero or one surviving alpha: if one, cap at max_alpha_concentration (implicit cash = remainder); return.
          5. Normalize surviving weights to sum=1.
          6. Iteratively cap any weight > max_alpha_concentration, redistribute overflow proportionally to uncapped weights;
             repeat until convergence or no uncapped receivers.
          7. Final normalization (preserve cap within numerical tolerance).
        """
        if not active_alphas:
            return {}
        usable = [a for a in active_alphas if regime in a['performance'] and a['performance'][regime].get('n_trades', 0) > 5]
        if not usable:
            regime = 'all'
            usable = [a for a in active_alphas if 'all' in a['performance'] and a['performance']['all'].get('n_trades', 0) > 5]
        if not usable:
            return {}
        returns_data = {a['id']: a['performance'][regime].get('returns', []) for a in usable}
        max_len = max(len(v) for v in returns_data.values())
        for k, v in returns_data.items():
            v.extend([0.0] * (max_len - len(v)))
        df = pd.DataFrame(returns_data)
        raw = self._inv_var_weights(df)
        floor = float(self.cfg.get('min_allocation_weight', 0.0))
        cap = float(self.cfg.get('max_alpha_concentration', 1.0))
        # Floor pass
        floored = {k: (w if w >= floor else 0.0) for k, w in raw.items()}
        non_zero = {k: w for k, w in floored.items() if w > 0.0}
        if not non_zero:
            return {}
        if len(non_zero) == 1:
            k = next(iter(non_zero.keys()))
            return {k: min(cap, 1.0)}  # remainder = cash implicit
        # Normalize initial
        s = sum(non_zero.values())
        weights = {k: w / s for k, w in non_zero.items()} if s > 0 else {}
        # Iterative cap & redistribute
        for _ in range(len(weights)):  # bounded iterations
            over = {k: w for k, w in weights.items() if w > cap + 1e-12}
            if not over:
                break
            overflow = sum(w - cap for w in over.values())
            # Cap the oversized weights
            for k in over:
                weights[k] = cap
            # Receivers: those strictly below cap after capping
            receivers = {k: w for k, w in weights.items() if w < cap - 1e-12}
            if not receivers or overflow <= 0:
                break  # cannot redistribute further; leave implicit cash remainder
            recv_sum = sum(receivers.values())
            if recv_sum <= 0:
                break
            for k, w in receivers.items():
                add = overflow * (w / recv_sum)
                weights[k] = w + add
        # Final normalization (do not violate cap; keep remainder as cash if total < 1 due to caps)
        total = sum(weights.values())
        if total > 0 and total > 1.0:  # scale down only if exceeded 1 (shouldn't normally if overflow redistributed)
            scale = 1.0 / total
            weights = {k: min(cap, w * scale) for k, w in weights.items()}
        return weights

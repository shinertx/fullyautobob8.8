import random, json, hashlib
import pandas as pd
from typing import List, Dict, Any
from v26meme.labs.simlab import SimLab

class FeatureProber:
    def __init__(self, *, fees_bps: int, slippage_bps: int, perturbations: int = 64, delta_fraction: float = 0.15, seed: int = 7, mdd_escalation_multiplier: float = 1.25):
        """Feature robustness prober applying random threshold & comparator perturbations.

        Args:
            fee_pct: Fee rate as a decimal fraction (e.g., 0.001 for 10 bps).
            slippage_pct: Slippage assumption as a decimal fraction.
            perturbations: Number of perturbation trials (k).
            delta_fraction: Fractional range for threshold perturbation (+/- d).
            seed: RNG seed for deterministic reproducibility (PIT‑safe).
            mdd_escalation_multiplier: Max allowed multiple of base MDD before counting
                a perturbation as fragile. Externalized to avoid magic number (was 1.25).
        """
        self.simlab = SimLab(fees_bps=fees_bps, slippage_bps=slippage_bps)
        self.k = int(perturbations); self.d = float(delta_fraction)
        self.mdd_mult = float(mdd_escalation_multiplier)
        random.seed(seed)

    def _leaves(self, formula):
        acc, q = [], [formula]
        while q:
            n = q.pop(0)
            if isinstance(n[0], list):
                q.append(n[0]); q.append(n[2])
            else:
                acc.append(n)
        return acc

    def score(self, df_features: pd.DataFrame, formula: List[Any]) -> Dict[str, Any]:
        stats_dict = self.simlab.run_backtest(df_features, formula)
        base = stats_dict.get('all', {})
        returns = base.get('returns') or []
        import numpy as _np
        if len(returns) == 0:
            return {"robust_score": 0.0, "base": {}}

        r = _np.asarray(returns, dtype=float)
        base_mean = float(r.mean())
        base_mdd = float(abs(_np.cumsum(r).min()))
        
        ok = 0
        deltas = []
        for _ in range(self.k):
            import copy
            f2 = copy.deepcopy(formula)
            leaves = self._leaves(f2)
            if not leaves: continue
            leaf = random.choice(leaves)
            if not isinstance(leaf[0], list):
                if random.random() < 0.8:
                    thr = float(leaf[2]); leaf[2] = thr * (1.0 + random.uniform(-self.d, self.d))
                else:
                    leaf[1] = ">" if leaf[1] == "<" else "<"
            
            res_stats = self.simlab.run_backtest(df_features, f2)
            res_r = _np.asarray(res_stats.get('all', {}).get('returns') or [], dtype=float)
            if res_r.size == 0:
                continue
            res_mean = float(res_r.mean())
            res_mdd = float(abs(_np.cumsum(res_r).min()))

            if (base_mean >= 0 and res_mean >= 0) or (base_mean < 0 and res_mean < 0):
                if res_mdd <= base_mdd * self.mdd_mult:
                    ok += 1
            
            deltas.append(res_mean - base_mean)
            
        base_stats = {
            "avg_return": base_mean,
            "mdd": base_mdd,
            "n_trades": int(r.size)
        }
        
        return {"robust_score": ok / float(self.k), "base": base_stats, "avg_return_deltas": deltas}

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any
from scipy.stats import ttest_1samp
from math import sqrt, log, comb

def _binomial_p_two_sided(wins: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test probability for observing a deviation at least as extreme.

    PIT: Uses only realized trade outcomes currently under evaluation; no future leakage.
    """
    if n <= 0:
        return 1.0
    exp = n * p
    dev = abs(wins - exp)
    pmf = [comb(n, k) * (p**k) * ((1-p)**(n-k)) for k in range(n+1)]
    prob = 0.0
    for k, mass in enumerate(pmf):
        if abs(k - exp) >= dev:
            prob += mass
    return float(min(1.0, max(0.0, prob)))

def purged_kfold_indices(n: int, k: int, embargo: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate purged K-fold indices with embargo.

    PIT: Ensures no look-ahead by purging an embargo window around each test fold.
    Parameters are adaptive knobs (k folds, embargo size) — avoid magic numbers.
    """
    folds = []
    fold_size = max(1, n // k)
    for i in range(k):
        start = i * fold_size
        end = n if i == k-1 else min(n, (i+1)*fold_size)
        test_idx = np.arange(start, end)
        train_mask = np.ones(n, dtype=bool)
        emb_lo = max(0, start - embargo)
        emb_hi = min(n, end + embargo)
        train_mask[emb_lo:emb_hi] = False
        folds.append((np.where(train_mask)[0], test_idx))
    return folds

def benjamini_hochberg(pvals: List[float], alpha: float):
    """Benjamini-Hochberg FDR control.

    PIT: Uses only provided p-values, no future leakage.
    """
    if not pvals:
        return [], 0.0
    m = len(pvals)
    order = np.argsort(pvals)
    thresh = 0.0
    accept = [False]*m
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= (rank/m)*alpha:
            accept[idx] = True
            thresh = max(thresh, pvals[idx])
    return accept, thresh

def _bootstrap_p_value(returns: pd.Series, n_iter: int, seed: int, method: str = "basic") -> float:
    """Bootstrap p-value (one-sided H1: mean > 0) using deterministic RNG.

    PIT: Operates only on realized return sample; deterministic seed → reproducible.
    Parameters (adaptive knobs):
      n_iter : number of bootstrap resamples.
      method : 'basic' or 'studentized' (placeholder; both use mean-centering to form null).
    """
    r = returns.dropna().astype(float)
    if r.empty or n_iter <= 0:
        return 1.0
    n = len(r)
    if n < 3 or r.std(ddof=1) == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    obs_mean = float(r.mean())
    # Null: mean = 0 → center sample
    centered = r - obs_mean
    means = []
    for _ in range(n_iter):
        arr = np.asarray(centered.values, dtype=float)
        sample = rng.choice(arr, size=n, replace=True)
        means.append(sample.mean())
    means_arr = np.asarray(means, dtype=float)
    # One-sided p-value: probability (mean >= obs_mean) under null
    # Under centering, obs_mean corresponds to shift 0; so use tail of bootstrap means > 0
    tail = float((means_arr >= obs_mean).mean()) if obs_mean > 0 else 1.0
    p = tail if tail > 0 else 1.0 / n_iter
    return float(min(1.0, max(0.0, p)))

def panel_cv_stats(panel_returns: Dict[str, pd.Series],
                   k_folds: int,
                   embargo: int,
                   alpha_fdr: float,
                   cv_method: str = "kfold",
                   sparse_fallback: Optional[Dict[str, Union[int,float]]] = None,
                   bootstrap: Optional[Dict[str, Union[int,float,str,bool]]] = None) -> Dict[str, Union[float,int,bool]]:
    """Aggregate panel cross-validation statistics with purged / combinatorial folds.

    Enhancements:
      - Optional bootstrap p-value path when trade count below bootstrap['min_trades'].
      - Retains sparse win-rate binomial fallback logic.
    All operations PIT-safe (closed-bar realized returns only).
    """
    all_returns_list = [s for s in panel_returns.values() if s is not None and not s.empty]
    if not all_returns_list:
        return {"p_value": 1.0, "mean_oos": 0.0, "sharpe_oos": 0.0, "total_trades": 0, "is_fallback": True, "n": 0}

    all_returns = pd.concat(all_returns_list).sort_index()
    n = len(all_returns)
    
    if n < k_folds and len(panel_returns) <= 1:
        # Not enough data for even one fold, use fallback for the whole series
        p_final = 1.0
        if n > 0:
            sfb = sparse_fallback or {}
            min_trades_for_ttest = sfb.get("min_trades_for_ttest", 5)
            if n < min_trades_for_ttest:
                wins = (all_returns > 0).sum()
                p_final = _binomial_p_two_sided(wins, n)
            else:
                t_stat, p_val = ttest_1samp(all_returns, 0)
                p_final = p_val / 2.0 if t_stat > 0 else 1.0 - (p_val / 2.0)
        
        mean_ret = float(all_returns.mean()) if not all_returns.empty else 0.0
        std_ret = float(all_returns.std()) if not all_returns.empty and all_returns.std() > 0 else 1.0
        return {
            "p_value": p_final,
            "mean_oos": mean_ret,
            "sharpe_oos": mean_ret / std_ret,
            "total_trades": n,
            "is_fallback": True,
        }

    # If multiple symbols are present, apply a simple CPCV-style symbol K-fold
    if len(panel_returns) > 1:
        symbols = [k for k, s in panel_returns.items() if s is not None and not s.empty]
        if not symbols:
            return {"p_value": 1.0, "mean_oos": 0.0, "sharpe_oos": 0.0, "total_trades": 0, "is_fallback": True, "n": 0}
        k_sym = max(2, min(k_folds, len(symbols)))
        oos_p_values_sym: List[float] = []
        total_oos_trades = 0
        for i in range(k_sym):
            test_syms = symbols[i::k_sym]
            if not test_syms:
                continue
            test_series = [panel_returns[sym].dropna() for sym in test_syms if sym in panel_returns]
            if not test_series:
                continue
            test_concat = pd.concat(test_series)
            total_oos_trades += len(test_concat)
            if len(test_concat) >= 2:
                t_stat, p_val = ttest_1samp(test_concat, 0)
                p_val = p_val / 2.0 if t_stat > 0 else 1.0 - (p_val / 2.0)
                oos_p_values_sym.append(p_val)
        p_final = float(np.mean(oos_p_values_sym)) if oos_p_values_sym else 1.0
        return {
            "p_value": p_final,
            "mean_oos": float(np.mean([s.mean() for s in all_returns_list])) if all_returns_list else 0.0,
            "sharpe_oos": float(np.mean([s.mean() / s.std(ddof=1) for s in all_returns_list if s.std(ddof=1) > 0])) if all_returns_list else 0.0,
            "total_trades": int(total_oos_trades),
            "is_fallback": False,
            "n": int(total_oos_trades),
        }

    indices = purged_kfold_indices(n=n, k=k_folds, embargo=embargo)
    
    oos_p_values = []
    oos_means = []
    oos_sharpes = []
    total_oos_trades = 0

    for train_idx, test_idx in indices:
        # train_returns = all_returns.iloc[train_idx].dropna() # train data not used for p-value
        test_returns = all_returns.iloc[test_idx].dropna()
        total_oos_trades += len(test_returns)

        if len(test_returns) >= 2:
            t_stat, p_val = ttest_1samp(test_returns, 0)
            # One-sided test: H1 is that mean > 0
            p_val = p_val / 2.0 if t_stat > 0 else 1.0 - (p_val / 2.0)
            oos_p_values.append(p_val)
            oos_means.append(test_returns.mean())
            oos_sharpes.append(test_returns.mean() / test_returns.std() if test_returns.std() > 0 else 0)

    # Fallback for sparse trade series where CV is not meaningful
    if not oos_p_values:
        sfb = sparse_fallback or {}
        min_trades_for_ttest = sfb.get("min_trades_for_ttest", 5)
        if len(all_returns) < min_trades_for_ttest:
            wins = (all_returns > 0).sum()
            p_final = _binomial_p_two_sided(wins, len(all_returns))
        else:
            # Bootstrap p-value for series that are too short for CV but not extremely sparse
            if bootstrap and bootstrap.get("enabled"):
                p_final = _bootstrap_p_value(
                    all_returns,
                    n_iter=int(bootstrap.get("n_iter", 500)),
                    seed=int(bootstrap.get("seed", 1337)),
                    method=str(bootstrap.get("method", "basic")),
                )
            else:
                # Default to t-test on full sample if bootstrap is off
                t_stat, p_val = ttest_1samp(all_returns, 0)
                p_final = p_val / 2.0 if t_stat > 0 else 1.0 - (p_val / 2.0)

        mean_ret = float(all_returns.mean()) if not all_returns.empty else 0.0
        std_ret = float(all_returns.std()) if not all_returns.empty and all_returns.std() > 0 else 1.0
        return {
            "p_value": p_final,
            "mean_oos": mean_ret,
            "sharpe_oos": mean_ret / std_ret,
            "total_trades": len(all_returns),
            "is_fallback": True,
            "n": len(all_returns),
        }

    p_final = float(np.mean(oos_p_values)) if oos_p_values else 1.0
    return {
        "p_value": p_final,
        "mean_oos": float(np.mean(oos_means)) if oos_means else 0.0,
        "sharpe_oos": float(np.mean(oos_sharpes)) if oos_sharpes else 0.0,
        "total_trades": total_oos_trades,
        "is_fallback": False,
        "n": int(total_oos_trades),
    }

def deflated_sharpe_ratio(returns: pd.Series, n_trials: int, sr_benchmark: float = 0.0) -> float:
    """Approximate Deflated Sharpe Ratio probability (Bailey & Lopez de Prado).

    PIT: Uses only historical returns array; n_trials reflects selection breadth.
    Returns probability Sharpe > benchmark after deflation for multiple testing.
    """
    r = returns.dropna()
    if r.empty or r.std(ddof=1) == 0:
        return 0.0
    sr_hat = float(r.mean() / r.std(ddof=1))
    n = len(r)
    infl = sqrt(max(0.0, 2.0 * log(max(2, n_trials)))) / sqrt(max(1, n - 1))
    sr_star = sr_benchmark + infl
    z = (sr_hat - sr_star) * sqrt(max(1, n - 1))
    from math import erf
    phi = 0.5 * (1.0 + erf(z / sqrt(2.0)))
    return max(0.0, min(1.0, float(phi)))

def compute_pbo(per_symbol_perf: Dict[str, Dict[str, float]], *, num_splits: int, min_symbols: int, seed: int) -> float:
    """Compute Probability of Backtest Overfitting (PBO) via symbol-based design/test splits.

    Method:
      - For each split, randomly (deterministically seeded) partition symbols into IS and OOS (≈50/50).
      - Rank formulas by mean IS performance; select top IS formula.
      - Compute its percentile rank in OOS distribution (higher mean = better rank).
      - Count an overfit event if OOS percentile (descending) is worse than median (i.e. rank position > n/2).
      - PBO = overfit_events / valid_splits.

    PIT: Uses only realized per-symbol aggregated returns (no forward data). Deterministic RNG seeded by provided seed.

    Args:
        per_symbol_perf: Mapping fid -> {symbol -> mean_return}.
        num_splits: Number of random symbol splits.
        min_symbols: Minimum symbols required to attempt PBO.
        seed: Deterministic seed base.

    Returns:
        PBO in [0,1]. Returns 0.0 if insufficient data.
    """
    # Collect global symbol set
    all_symbols = set()
    for perf in per_symbol_perf.values():
        all_symbols.update(perf.keys())
    symbols = sorted(all_symbols)
    if len(symbols) < max(2, min_symbols):
        return 0.0
    import random as _r
    _r.seed(seed)
    overfit = 0
    valid = 0
    n_formulas = len(per_symbol_perf)
    if n_formulas < 2:
        return 0.0
    for split_idx in range(num_splits):
        # 50/50 split (ceil for IS)
        _r.shuffle(symbols)
        half = max(1, len(symbols)//2)
        is_syms = set(symbols[:half])
        oos_syms = set(symbols[half:])
        if not oos_syms:
            continue
        # Compute IS/OOS means per formula
        is_means: Dict[str, float] = {}
        oos_means: Dict[str, float] = {}
        for fid, sym_map in per_symbol_perf.items():
            is_vals = [v for s, v in sym_map.items() if s in is_syms]
            oos_vals = [v for s, v in sym_map.items() if s in oos_syms]
            if is_vals:
                is_means[fid] = float(sum(is_vals)/len(is_vals))
            if oos_vals:
                oos_means[fid] = float(sum(oos_vals)/len(oos_vals))
        if len(is_means) < 2:
            continue
        # Select best IS formula
        best_fid = max(is_means.items(), key=lambda kv: kv[1])[0]
        # Need OOS distribution including best
        if best_fid not in oos_means or len(oos_means) < 2:
            continue
        # Rank OOS (descending higher better)
        ranked = sorted(oos_means.items(), key=lambda kv: kv[1], reverse=True)
        positions = {fid: idx for idx, (fid, _) in enumerate(ranked)}
        pos = positions[best_fid]
        # percentile rank (0 = best, 1 = worst)
        percentile = pos / (len(ranked)-1) if len(ranked) > 1 else 0.0
        if percentile > 0.5:
            overfit += 1
        valid += 1
    if valid == 0:
        return 0.0
    return float(overfit / valid)

class Validator:
    """
    Encapsulates the full alpha validation pipeline, including cross-validation,
    False Discovery Rate control, and Deflated Sharpe Ratio analysis.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cv_folds = int(config.get('cv_folds', 5))
        self.cv_embargo = int(config.get('cv_embargo_bars', 5))
        self.fdr_alpha = float(config.get('fdr_alpha', 0.1))
        self.dsr_cfg = config.get('dsr', {})
        self.dsr_enabled = bool(self.dsr_cfg.get('enabled', True))
        self.dsr_bench = float(self.dsr_cfg.get('benchmark_sr', 0.0))
        self.dsr_min_prob = float(self.dsr_cfg.get('min_prob', 0.5))
        self.bootstrap_cfg = config.get('bootstrap', {})
        self.min_trades_gate = int(config.get('min_trades', 20))

    def validate_batch(self, evaluated_alphas: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
        """
        Applies the full validation suite to a batch of evaluated alpha candidates.
        Returns a tuple: (list_of_survivors, dict_of_rejections).
        """
        if not evaluated_alphas:
            return [], {}

        rejections: Dict[str, List[str]] = {
            "min_trades": [], "fdr": [], "dsr": []
        }
        
        pvals, fids, formula_stats = [], [], {}

        for alpha in evaluated_alphas:
            fid = alpha.get('alpha_id')
            trades_df = alpha.get('trades')
            per_symbol_map = alpha.get('per_symbol') if isinstance(alpha.get('per_symbol'), dict) else None
            
            if (trades_df is None or trades_df.empty or 'pnl' not in trades_df) and not per_symbol_map:
                if fid: rejections["min_trades"].append(fid)
                continue

            total_trades = len(trades_df) if (trades_df is not None and 'pnl' in trades_df) else sum(len(v or []) for v in per_symbol_map.values()) if per_symbol_map else 0
            if total_trades < self.min_trades_gate:
                if fid: rejections["min_trades"].append(fid)
                continue

            if per_symbol_map:
                # Build symbol-wise series for CPCV
                panel_returns = {str(sym): pd.Series(vals, dtype=float) for sym, vals in per_symbol_map.items() if vals}
            else:
                panel_returns = {'panel': trades_df['pnl']}
            
            cv_res = panel_cv_stats(
                panel_returns,
                k_folds=self.cv_folds, embargo=self.cv_embargo,
                alpha_fdr=self.fdr_alpha, bootstrap=self.bootstrap_cfg
            )
            
            p_val = float(cv_res.get('p_value', 1.0))
            
            all_returns = trades_df['pnl']
            dsr_prob = deflated_sharpe_ratio(all_returns, n_trials=len(evaluated_alphas), sr_benchmark=self.dsr_bench) if self.dsr_enabled else 1.0
            
            if fid:
                pvals.append(p_val)
                fids.append(fid)
                formula_stats[fid] = {
                    'p_value': p_val, 'dsr_prob': dsr_prob, 'n_trades': total_trades,
                    'sharpe': alpha.get('sharpe'), 'formula': alpha.get('formula'),
                    'returns': all_returns.tolist()
                }

        if not pvals:
            return [], rejections

        flags, _ = benjamini_hochberg(pvals, self.fdr_alpha)
        
        final_survivors = []
        for fid, flag in zip(fids, flags):
            stats = formula_stats.get(fid, {})
            
            if not flag:
                rejections["fdr"].append(fid)
                continue

            if self.dsr_enabled and stats.get('dsr_prob', 0.0) < self.dsr_min_prob:
                rejections["dsr"].append(fid)
                continue
            
            survivor = {
                'alpha_id': fid, # Changed 'fid' to 'alpha_id' to match hyper_lab
                'formula': stats.get('formula'),
                'sharpe': stats.get('sharpe', 0.0),
                'p_value': stats.get('p_value', 1.0),
                'dsr_prob': stats.get('dsr_prob', 0.0),
                'n_trades': stats.get('n_trades', 0),
                'returns': stats.get('returns', []),
                'universe': self.config.get('universe', []),
                'timeframe': self.config.get('timeframe', '1h'),
            }
            final_survivors.append(survivor)
            
        return final_survivors, rejections

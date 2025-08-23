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
                   bootstrap: Optional[Dict[str, Union[int,float,str,bool]]] = None) -> Dict[str, Union[float,int]]:
    """Aggregate panel cross-validation statistics with purged / combinatorial folds.

    Enhancements:
      - Optional bootstrap p-value path when trade count below bootstrap['min_trades'].
      - Retains sparse win-rate binomial fallback logic.
    All operations PIT-safe (closed-bar realized returns only).
    """
    all_oos: List[pd.Series] = []
    total_trades = 0
    wins = 0
    for _sym, s in panel_returns.items():
        s = s.dropna()
        if s.empty or len(s) < max(20, k_folds*5):
            continue
        n = len(s)
        total_trades += n
        wins += int((s > 0).sum())
        folds = purged_kfold_indices(n, k_folds, embargo)
        if cv_method and str(cv_method).lower() == "cpcv" and k_folds > 1:
            comb_folds = []
            for i in range(k_folds):
                for j in range(i+1, k_folds):
                    test_idx = np.unique(np.concatenate([folds[i][1], folds[j][1]]))
                    train_mask = np.ones(n, dtype=bool)
                    emb_lo_i = max(0, folds[i][1][0] - embargo); emb_hi_i = min(n, folds[i][1][-1] + 1 + embargo)
                    train_mask[emb_lo_i:emb_hi_i] = False
                    emb_lo_j = max(0, folds[j][1][0] - embargo); emb_hi_j = min(n, folds[j][1][-1] + 1 + embargo)
                    train_mask[emb_lo_j:emb_hi_j] = False
                    train_mask[test_idx] = False
                    comb_folds.append((np.where(train_mask)[0], test_idx))
            folds_to_use = comb_folds
        else:
            folds_to_use = folds
        oos_chunks = [s.iloc[test_idx] for _, test_idx in folds_to_use]
        if oos_chunks:
            all_oos.append(pd.concat(oos_chunks))
    if not all_oos:
        return {"p_value": 1.0, "mean_oos": 0.0, "n": 0}
    all_oos_concat = pd.concat(all_oos)
    # Normalize to Series of floats deterministically
    if isinstance(all_oos_concat, pd.DataFrame):
        if all_oos_concat.shape[1] == 1:
            all_oos_concat = all_oos_concat.iloc[:, 0]  # type: ignore[index]
        else:
            all_oos_concat = all_oos_concat.mean(axis=1)  # type: ignore[arg-type]
    if not isinstance(all_oos_concat, pd.Series):  # final guard
        try:
            all_oos_concat = pd.Series(all_oos_concat)
        except Exception:
            return {"p_value": 1.0, "mean_oos": 0.0, "n": 0}
    all_oos_concat = pd.to_numeric(all_oos_concat, errors='coerce').dropna()
    if all_oos_concat.empty:
        return {"p_value": 1.0, "mean_oos": 0.0, "n": 0}
    n_total = int(all_oos_concat.shape[0])
    std_val = float(all_oos_concat.std(ddof=1)) if n_total > 1 else 0.0

    # Decide p-value computation path
    p: float = 1.0
    use_bootstrap = False
    if bootstrap and bool(bootstrap.get('enabled', True)):
        min_bt = int(bootstrap.get('min_trades', 60))
        if n_total < min_bt:
            use_bootstrap = True
    if std_val == 0.0 or n_total < 5:
        p = 1.0
    elif use_bootstrap and bootstrap:
        n_iter = int(bootstrap.get('n_iter', 500))
        seed = int(bootstrap.get('seed', 1337))
        method = str(bootstrap.get('method', 'basic'))
        p = _bootstrap_p_value(all_oos_concat, n_iter=n_iter, seed=seed, method=method)
    else:
        # One-sided t-test (greater than 0 mean)
        stat_val, p_raw = ttest_1samp(all_oos_concat, popmean=0.0, alternative='greater')
        try:
            if isinstance(p_raw, (float, int, np.floating, np.integer)):
                p = float(p_raw)
            elif hasattr(p_raw, 'shape') and getattr(p_raw, 'size', 0) > 0:
                p = float(np.asarray(p_raw).flat[0])
            elif isinstance(p_raw, (list, tuple)) and p_raw:
                p = float(p_raw[0])
            else:
                p = 1.0
        except Exception:
            p = 1.0

    # Sparse fallback (win-rate binomial) if configured
    if sparse_fallback and total_trades > 0:
        min_tr = int(sparse_fallback.get('min_trades', 25))
        p_gate = sparse_fallback.get('p_gate', 0.95)
        try:
            p_gate_f = float(p_gate) if not isinstance(p_gate, bool) else 0.95
            if 1 < p_gate_f <= 100:
                p_gate_f /= 100.0
        except Exception:
            p_gate_f = 0.95
        if total_trades < min_tr and p >= p_gate_f:
            p_bin = _binomial_p_two_sided(wins, total_trades, p=0.5)
            p = min(p, p_bin)

    mean_val = float(all_oos_concat.mean()) if n_total else 0.0
    return {"p_value": p, "mean_oos": mean_val, "n": n_total}

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

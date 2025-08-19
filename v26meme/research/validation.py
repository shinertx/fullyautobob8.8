import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from scipy.stats import ttest_1samp

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

def panel_cv_stats(panel_returns: Dict[str, pd.Series], k_folds: int, embargo: int, alpha_fdr: float, cv_method: str = "kfold"):
    """Aggregate panel cross-validation statistics with purged K-fold and FDR gate helper.

    PIT: Operates strictly on historical returns; ensures variance guard.
    """
    all_oos = []
    for _sym, s in panel_returns.items():
        s = s.dropna()
        if s.empty or len(s) < max(20, k_folds*5): continue
        n = len(s)
        folds = purged_kfold_indices(n, k_folds, embargo)
        if cv_method and str(cv_method).lower() == "cpcv" and k_folds > 1:
            # Combinatorial Purged CV: generate combinations of test folds (e.g. pairs) as additional OOS scenarios
            comb_folds = []
            for i in range(k_folds):
                for j in range(i+1, k_folds):
                    test_idx = np.concatenate([folds[i][1], folds[j][1]])
                    test_idx = np.unique(test_idx)
                    test_idx.sort()
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
        if oos_chunks: all_oos.append(pd.concat(oos_chunks))
    if not all_oos:
        return {"p_value": 1.0, "mean_oos": 0.0, "n": 0}
    all_oos_concat = pd.concat(all_oos)
    if isinstance(all_oos_concat, pd.DataFrame):
        all_oos_concat = all_oos_concat.iloc[:,0]
    all_oos_concat = all_oos_concat.astype(float)
    std_val = float(all_oos_concat.std(ddof=1)) if all_oos_concat.shape[0] > 1 else 0.0
    if std_val == 0.0 or len(all_oos_concat) < 10:
        p = 1.0
    else:
        stat_val, p_raw = ttest_1samp(all_oos_concat, popmean=0.0, alternative='greater')
        try:  # p_raw is scalar float from scipy, cast defensively
            p = float(p_raw) if p_raw is not None else 1.0  # type: ignore[arg-type]
        except Exception:
            p = 1.0
    mean_val = float(all_oos_concat.mean()) if len(all_oos_concat) else 0.0
    return {"p_value": p, "mean_oos": mean_val, "n": int(all_oos_concat.shape[0])}

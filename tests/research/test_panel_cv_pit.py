import pandas as pd, numpy as np
from v26meme.research.validation import purged_kfold_indices, panel_cv_stats


def test_purged_kfold_no_leakage():
    n=120; k=4; embargo=2
    folds = purged_kfold_indices(n, k, embargo)
    # Ensure embargo gap between train/test
    for train_idx, test_idx in folds:
        if len(test_idx)==0: continue
        lo, hi = test_idx.min(), test_idx.max()
        # No test index should appear in train
        assert set(test_idx).isdisjoint(set(train_idx))
        # Embargo zone removed
        emb_lo = max(0, lo-embargo); emb_hi = min(n, hi+embargo+1)
        train_set = set(train_idx)
        for e in range(emb_lo, emb_hi):
            assert e not in train_set


def test_panel_cv_stats_basic():
    # Create synthetic positive mean returns series for two symbols
    np.random.seed(1337)
    r1 = pd.Series(np.random.normal(0.001, 0.01, 300))
    r2 = pd.Series(np.random.normal(0.0005, 0.01, 300))
    stats = panel_cv_stats({"A": r1, "B": r2}, k_folds=4, embargo=2, alpha_fdr=0.1)
    assert stats["n"] > 0
    assert -0.01 < stats["mean_oos"] < 0.01  # mean in plausible band


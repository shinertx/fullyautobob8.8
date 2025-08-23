import pandas as pd, numpy as np
from v26meme.research.validation import panel_cv_stats


def test_sparse_fallback_binomial_lowers_pvalue():
    np.random.seed(0)
    r = pd.Series(np.random.normal(0.001, 0.001, 8))  # small N to trigger sparse
    panel = {"SYM": r}
    stats_fb = panel_cv_stats(panel, k_folds=4, embargo=1, alpha_fdr=0.2, cv_method='kfold',
                              sparse_fallback={'min_trades': 20, 'p_gate': 0.95})
    stats_no_fb = panel_cv_stats(panel, k_folds=4, embargo=1, alpha_fdr=0.2, cv_method='kfold',
                                 sparse_fallback={'min_trades': 0, 'p_gate': 0.95})
    assert stats_no_fb['p_value'] >= stats_fb['p_value']

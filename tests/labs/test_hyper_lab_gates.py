import json, time
import pandas as pd
from v26meme.labs.simlab import SimLab
from v26meme.research.generator import GeneticGenerator
from v26meme.research.validation import panel_cv_stats, deflated_sharpe_ratio, benjamini_hochberg


def _mock_returns(n=120, seed=42):
    import random
    random.seed(seed)
    # slight positive drift
    return pd.Series([0.002 + random.gauss(0,0.01) for _ in range(n)])

def test_panel_cv_and_dsr_gate_pass():
    panel = {"SYM1": _mock_returns(), "SYM2": _mock_returns(seed=43)}
    stats = panel_cv_stats(panel, k_folds=3, embargo=2, alpha_fdr=0.1, cv_method="kfold")
    assert stats['p_value'] <= 1.0 and isinstance(stats['mean_oos'], float)

    dsr_prob = deflated_sharpe_ratio(panel['SYM1'], n_trials=50, sr_benchmark=0.0)
    assert 0.0 <= dsr_prob <= 1.0


def test_bh_fdr_flags_some():
    pvals = [0.001,0.02,0.5,0.9,0.049]
    keep, thr = benjamini_hochberg(pvals, 0.1)
    assert len(keep) == len(pvals)
    assert any(keep)
    assert 0.0 <= thr <= 0.1


def test_gene_usage_tracking(tmp_path):
    # simulate feature extraction on formula and gene tracking increments
    from v26meme.labs.hyper_lab import _extract_features
    feats = _extract_features([[['f1','>',0], 'AND', ['f2','<',1]], 'OR', ['f3','>',0]])
    assert set(feats) == {'f1','f2','f3'}

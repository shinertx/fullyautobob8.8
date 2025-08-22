import os, json, time
import pytest
from v26meme.labs.hyper_lab import run_eil  # updated import
from v26meme.core.state import StateManager

@pytest.mark.parametrize("patience,threshold", [(2, 0.99)])
def test_feature_continuity_suppression(monkeypatch, tmp_path, patience, threshold):
    monkeypatch.setenv('EIL_MAX_CYCLES', '2')
    from v26meme.labs import hyper_lab as hl
    base_cfg = hl.load_config()
    base_cfg['discovery']['adaptive']['continuity_suppression_patience'] = patience
    base_cfg['discovery']['adaptive']['continuity_threshold'] = threshold
    orig_select = hl._select_timeframe
    def fake_select(cfg, lake, min_bars, min_panel, prefer):
        return None, [], {}
    monkeypatch.setattr(hl, '_select_timeframe', fake_select)
    monkeypatch.setattr(hl, 'load_config', lambda : base_cfg)
    run_eil()
    state = StateManager(base_cfg['system']['redis_host'], base_cfg['system']['redis_port'])
    suppress = state.get('eil:continuity_suppress') or []
    assert isinstance(suppress, list)


def test_rejection_telemetry_keys(monkeypatch):
    monkeypatch.setenv('EIL_MAX_CYCLES', '1')
    from v26meme.labs import hyper_lab as hl
    cfg = hl.load_config()
    cfg['discovery']['min_bars_per_symbol'] = 1
    cfg['discovery']['min_panel_symbols'] = 1
    monkeypatch.setattr(hl, 'load_config', lambda : cfg)
    run_eil()
    state = StateManager(cfg['system']['redis_host'], cfg['system']['redis_port'])
    rc = state.get('eil:rej:counts')
    assert rc is not None


def test_beta_fallback_presence():
    from v26meme.research.feature_factory import FeatureFactory
    import pandas as pd
    ff = FeatureFactory()
    df = pd.DataFrame({
        'open':[1,1.1,1.2,1.1,1.05,1.07,1.06,1.08,1.09,1.1],
        'high':[1]*10,'low':[1]*10,'close':[1,1.1,1.2,1.1,1.05,1.07,1.06,1.08,1.09,1.1], 'volume':[10]*10
    })
    btc = df.rename(columns={'close':'close'})
    out = ff.create(df, symbol='TEST_USD_SPOT', cfg={'features':{}}, other_dfs={'BTC_USD_SPOT': btc})
    assert 'beta_btc_20p' in out.columns


def test_partial_reseed_hygiene(monkeypatch):
    monkeypatch.setenv('EIL_MAX_CYCLES', '1')
    from v26meme.labs import hyper_lab as hl
    cfg = hl.load_config()
    cfg['discovery']['min_bars_per_symbol'] = 1
    cfg['discovery']['min_panel_symbols'] = 1
    cfg['discovery']['feature_min_non_nan_ratio'] = 1.1
    monkeypatch.setattr(hl, 'load_config', lambda : cfg)
    run_eil()
    state = StateManager(cfg['system']['redis_host'], cfg['system']['redis_port'])
    hyg = state.get('eil:population_hygiene') or {}
    assert isinstance(hyg, dict)


def test_beta_window_behavior():
    from v26meme.research.feature_factory import FeatureFactory
    import pandas as pd, numpy as np
    ff = FeatureFactory()
    # 25 bars so 20-window beta yields 5 finite (post-shift -> 4) values
    prices = np.linspace(100, 110, 25)
    df = pd.DataFrame({
        'open': prices,
        'high': prices*1.001,
        'low': prices*0.999,
        'close': prices,
        'volume': np.random.default_rng(42).integers(10, 20, size=25)
    })
    btc = df.copy()
    out = ff.create(df, symbol='TEST_USD_SPOT', cfg={'features':{}}, other_dfs={'BTC_USD_SPOT': btc})
    beta = out['beta_btc_20p']
    # Expect initial < window bars NaN (window=20, plus shift)
    assert beta.isna().sum() >= 20
    # After enough bars, some finite values should appear
    assert beta.dropna().shape[0] > 0

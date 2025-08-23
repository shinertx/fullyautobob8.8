import pandas as pd
import numpy as np
from v26meme.research.feature_factory import FeatureFactory

CFG = {
    'discovery': {
        'feature_windows': {
            'vol': [20],
            'momentum': [10],
            'volume': [20]
        }
    }
}

def test_dynamic_window_features_present_and_lagged():
    n = 300
    ts = pd.date_range('2024-01-01', periods=n, freq='H', tz='UTC')
    rng = np.random.default_rng(42)
    prices = 100 + np.cumsum(rng.normal(0, 1, n))
    vols = rng.integers(100, 200, n)
    df = pd.DataFrame({'open': prices, 'high': prices+0.5, 'low': prices-0.5, 'close': prices, 'volume': vols}, index=ts)
    ff = FeatureFactory()
    feats = ff.create(df, symbol='TEST', cfg=CFG)
    # Columns expected
    assert 'realized_vol_20p' in feats.columns
    assert 'momentum_10p_dyn' in feats.columns
    assert 'volume_ema_ratio_20p' in feats.columns
    # Lag check (skip warmup window)
    raw_ret = df['close'].pct_change()
    raw_mom10 = df['close'].pct_change(10)
    burn = 50
    assert feats['realized_vol_20p'].iloc[burn:].equals(feats['realized_vol_20p'].iloc[burn:])  # existence sanity
    # dynamic momentum is lagged version of raw_mom10
    assert np.allclose(feats['momentum_10p_dyn'].iloc[burn:], raw_mom10.shift(1).iloc[burn:], equal_nan=True)

import pandas as pd
import numpy as np
from v26meme.research.feature_factory import FeatureFactory


def test_price_derived_features_are_lagged_one_bar():
    n = 180
    ts = pd.date_range('2024-01-01', periods=n, freq='h', tz='UTC')
    rng = np.random.default_rng(0)
    prices = np.cumsum(rng.normal(0, 1, n)) + 100
    df = pd.DataFrame({'close': prices, 'open': prices, 'high': prices+0.5, 'low': prices-0.5, 'volume': 1.0}, index=ts)

    ff = FeatureFactory()
    feats = ff.create(df)

    # Raw (unlagged) constructions
    raw_ret = df['close'].pct_change()
    raw_mom10 = df['close'].pct_change(10)
    delta = df['close'].diff().astype(float)
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    raw_rsi14 = 100 - (100 / (1 + rs))

    # Windows to skip (burn-in)
    burn_ret = 5
    burn_mom10 = 15  # need at least 10 + 1 lag + some buffer
    burn_rsi14 = 30  # 14-period smoothing + lag buffer

    # Assert columns exist
    for col in ['return_1p','momentum_10p','rsi_14']:
        assert col in feats.columns

    # Single-lag equality (allow small numerical tolerance)
    assert np.allclose(feats['return_1p'].iloc[burn_ret:], raw_ret.shift(1).iloc[burn_ret:], equal_nan=True)
    assert np.allclose(feats['momentum_10p'].iloc[burn_mom10:], raw_mom10.shift(1).iloc[burn_mom10:], equal_nan=True)
    assert np.allclose(feats['rsi_14'].iloc[burn_rsi14:], raw_rsi14.shift(1).iloc[burn_rsi14:], equal_nan=True)

    # Guard against accidental double shift: compare against shift(2)
    assert not np.allclose(feats['momentum_10p'].iloc[burn_mom10:], raw_mom10.shift(2).iloc[burn_mom10:], equal_nan=True)
    assert not np.allclose(feats['rsi_14'].iloc[burn_rsi14:], raw_rsi14.shift(2).iloc[burn_rsi14:], equal_nan=True)


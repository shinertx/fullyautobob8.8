import pandas as pd, numpy as np
from v26meme.research.feature_prober import FeatureProber
from v26meme.labs.simlab import SimLab

# Minimal synthetic feature + price series to exercise robustness score deterministically

def _make_df(n=240):
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")  # use lowercase 'h' to avoid deprecation warning
    price = 100 + np.cumsum(np.random.normal(0, 0.5, n))
    df = pd.DataFrame({
        'open': price + np.random.normal(0,0.1,n),
        'high': price + np.random.normal(0.2,0.1,n),
        'low': price - np.random.normal(0.2,0.1,n),
        'close': price,
        'volume': np.random.uniform(10,50,n)
    }, index=idx)
    # Simple engineered features consumed by simlab/feature logic
    df['return_1p'] = df['close'].pct_change().fillna(0)
    df['volatility_20p'] = df['return_1p'].rolling(20).std().fillna(0)
    df['momentum_10p'] = df['close'].pct_change(10).fillna(0)
    df['rsi_14'] = 50.0
    df['close_vs_sma50'] = (df['close']/df['close'].rolling(50).mean()).fillna(1.0)
    df['hod_sin'] = 0.0
    df['hod_cos'] = 1.0
    df['round_proximity'] = 0.0
    df['btc_corr_20p'] = 0.0
    df['eth_btc_ratio'] = 1.0
    return df


def test_feature_prober_scores():
    np.random.seed(1337)
    df = _make_df()
    formula = ["return_1p", ">", 0.0]
    prober = FeatureProber(fees_bps=10, slippage_bps=5, perturbations=8, delta_fraction=0.1, seed=42)
    res = prober.score(df, formula)
    assert 'robust_score' in res
    assert 0.0 <= res['robust_score'] <= 1.0

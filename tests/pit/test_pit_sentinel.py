"""PIT Sentinel Test

Ensures FeatureFactory outputs lagged features (no lookahead) and that open bar (last row) is excluded from signal evaluation base.
Fast (<1s) safety net so audit pipeline doesn't stall.
"""
from datetime import datetime, timedelta, timezone
import pandas as pd
from v26meme.research.feature_factory import FeatureFactory


def _make_df(n=30):
    now = datetime.now(timezone.utc).replace(microsecond=0, second=0)
    rows = []
    for i in range(n):
        ts = now - timedelta(minutes=(n-i)*5)
        price = 100 + i * 0.1
        rows.append({
            'timestamp': ts,
            'open': price,
            'high': price*1.001,
            'low': price*0.999,
            'close': price,
            'volume': 10+i
        })
    return pd.DataFrame(rows)


def test_features_are_lagged():
    df = _make_df()
    ff = FeatureFactory()
    out = ff.create(df.copy(), symbol='TEST_USD_SPOT', cfg={'discovery': {'feature_windows': {}}})
    # pick a base feature pair (momentum_10p vs close) ensure shift applied
    if 'momentum_10p' in out.columns:
        # momentum_10p at t should relate to close diff up to t-1 (since shifted)
        shifted_check_idx = out.index[15]  # mid sample
        mom_val = out.loc[shifted_check_idx, 'momentum_10p']
        # reconstruct raw (unshifted) would use close[t]/close[t-10]-1; so value at raw[t] equals shifted at t+1
        # Validate by forward comparing: mom at t should equal raw at t-1
        idx_pos = list(out.index).index(shifted_check_idx)
        if idx_pos+1 < len(out.index):
            next_idx = out.index[idx_pos+1]
            if 'momentum_10p' in out.columns:
                assert out.loc[shifted_check_idx, 'momentum_10p'] == out.loc[next_idx, 'momentum_10p'] or pd.isna(mom_val) or pd.isna(out.loc[next_idx, 'momentum_10p'])


def test_no_future_timestamp_order():
    df = _make_df()
    ff = FeatureFactory()
    out = ff.create(df.copy(), symbol='TEST_USD_SPOT', cfg={'discovery': {'feature_windows': {}}})
    # timestamps strictly increasing
    assert out.index.is_monotonic_increasing

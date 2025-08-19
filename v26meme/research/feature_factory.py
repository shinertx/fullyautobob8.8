import pandas as pd
import numpy as np
import math

class FeatureFactory:
    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in ['open','high','low','close','volume'] if c in df.columns]
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=cols)
        for c in cols:
            df = df[df[c].apply(lambda x: isinstance(x,(int,float)) and math.isfinite(float(x)))]
        return df

    def create(self, df: pd.DataFrame, symbol: str | None = None,
               cfg: dict | None = None, other_dfs: dict | None = None) -> pd.DataFrame:
        if df.empty: return df
        df = self._clean(df)
        out = df.reset_index().rename(columns={"index":"timestamp"})
        if "timestamp" not in out.columns: out["timestamp"] = out.index
        out = out.sort_values("timestamp").reset_index(drop=True)

        # Core features (PIT-safe)
        out['return_1p'] = out['close'].pct_change()
        out['volatility_20p'] = out['return_1p'].rolling(20).std() * np.sqrt(20)
        out['momentum_10p'] = out['close'].pct_change(10).shift(1)
        delta = out['close'].astype(float).diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss.replace(0, np.nan))
        out['rsi_14'] = 100 - (100 / (1 + rs))
        out['rsi_14'] = out['rsi_14'].shift(1)
        out['sma_50'] = out['close'].rolling(50).mean()
        out['close_vs_sma50'] = ((out['close'] - out['sma_50']) / out['sma_50']).shift(1)

        # Time-of-day (PIT)
        ts = pd.to_datetime(out['timestamp'], utc=True)
        out['hod_sin'] = np.sin(2 * np.pi * ts.dt.hour / 24.0)
        out['hod_cos'] = np.cos(2 * np.pi * ts.dt.hour / 24.0)

        # Round-number proximity (scale-aware, PIT)
        logp = np.log10(out['close'].clip(lower=1e-9))
        frac = logp - np.floor(logp)
        prox = np.minimum(np.abs(frac), np.abs(1 - frac))
        out['round_proximity'] = -prox

        # Cross-asset (lagged)
        if other_dfs:
            btc = other_dfs.get('BTC_USD_SPOT'); eth = other_dfs.get('ETH_USD_SPOT')
            if btc is not None and not btc.empty:
                btc_close = btc['close'].reindex(ts, method='pad')
                out['btc_corr_20p'] = out['close'].rolling(20).corr(btc_close).shift(1)
            if eth is not None and btc is not None and not eth.empty and not btc.empty:
                eth_close = eth['close'].reindex(ts, method='pad')
                btc_close = btc['close'].reindex(ts, method='pad')
                out['eth_btc_ratio'] = (eth_close / btc_close).shift(1)

        # Strict PIT: shift price-derived features so trading at bar t uses info up to t-1
        shift_cols = [
            'return_1p','volatility_20p','momentum_10p','rsi_14','sma_50','close_vs_sma50','round_proximity'
        ]
        for c in shift_cols:
            if c in out.columns:
                out[c] = out[c].shift(1)

        out = out.dropna()
        out = out.set_index('timestamp')
        return out

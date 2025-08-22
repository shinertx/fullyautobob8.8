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

    def _dedupe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if not df.columns.duplicated().any():
            return df
        ordered = {}
        for i, c in enumerate(df.columns):
            if c not in ordered:
                ordered[c] = df.iloc[:, i]
        # Reconstruct DataFrame with first occurrences only (preserves original index)
        return pd.DataFrame(ordered, index=df.index)

    def create(self, df: pd.DataFrame, symbol: str | None = None,
               cfg: dict | None = None, other_dfs: dict | None = None) -> pd.DataFrame:
        """Create PIT-safe feature set with robust timestamp de-duplication.
        Notes:
          - All features referencing contemporaneous prices are shifted by 1 bar to enforce PIT correctness.
          - Cross-asset beta uses a custom numpy rolling window (ddof=1) to avoid pandas internal dtype comparison warnings.
        """
        if df.empty:
            return df
        df = self._clean(df)
        df = self._dedupe_columns(df)

        idx_name = df.index.name
        has_col_ts = 'timestamp' in df.columns
        if has_col_ts and (idx_name is None or idx_name != 'timestamp'):
            out = df.reset_index(drop=True)
        elif not has_col_ts:
            out = df.reset_index()
            if 'index' in out.columns and 'timestamp' not in out.columns:
                out = out.rename(columns={'index': 'timestamp'})
        elif has_col_ts and idx_name == 'timestamp':
            out = df.reset_index(drop=True)
        else:
            out = df.reset_index()
            if 'index' in out.columns and 'timestamp' in out.columns:
                out = out.drop(columns=['index'])
            elif 'index' in out.columns:
                out = out.rename(columns={'index': 'timestamp'})

        if 'timestamp' not in out.columns:
            out['timestamp'] = out.index
        out = self._dedupe_columns(out)
        try:
            out['timestamp'] = pd.to_datetime(out['timestamp'], utc=True)
        except Exception:
            pass
        out = self._dedupe_columns(out)  # final pass before sort
        out = out.sort_values('timestamp').reset_index(drop=True)

        # --- Feature engineering (PIT) ---
        out['return_1p'] = out['close'].pct_change()
        out['volatility_20p'] = out['return_1p'].rolling(20).std() * np.sqrt(20)
        out['momentum_10p'] = out['close'].pct_change(10)
        delta = out['close'].astype(float).diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss.replace(0, np.nan))
        out['rsi_14'] = 100 - (100 / (1 + rs))
        out['sma_50'] = out['close'].rolling(50).mean()
        out['close_vs_sma50'] = (out['close'] - out['sma_50']) / out['sma_50']

        # Time-of-day cyclic encodings
        ts = pd.to_datetime(out['timestamp'], utc=True)
        out['hod_sin'] = np.sin(2 * np.pi * ts.dt.hour / 24.0)
        out['hod_cos'] = np.cos(2 * np.pi * ts.dt.hour / 24.0)

        # Round-number proximity
        logp = np.log10(out['close'].clip(lower=1e-9))
        frac = logp - np.floor(logp)
        prox = np.minimum(np.abs(frac), np.abs(1 - frac))
        out['round_proximity'] = -prox

        # Cross-asset (lagged inputs only)
        if other_dfs:
            def _align_close(ext_df):
                if ext_df is None or ext_df.empty or 'close' not in ext_df.columns:
                    return None
                ser = ext_df['close']
                if not isinstance(ser.index, pd.DatetimeIndex):
                    if len(ser) == len(ts):
                        ser.index = ts
                    else:
                        base_index = ts[:len(ser)] if len(ts) >= len(ser) else ts
                        ser = pd.Series(ser.values[:len(base_index)], index=base_index)
                return ser.reindex(ts, method='pad')
            btc = other_dfs.get('BTC_USD_SPOT'); eth = other_dfs.get('ETH_USD_SPOT')
            btc_close = _align_close(btc)
            if btc_close is not None:
                out['btc_corr_20p'] = out['close'].rolling(20).corr(btc_close)
                ret_sym = out['close'].pct_change().astype(float)
                ret_btc = btc_close.pct_change().astype(float)
                ret_sym = ret_sym.replace([np.inf, -np.inf], np.nan)
                ret_btc = ret_btc.replace([np.inf, -np.inf], np.nan)
                window = 20
                rs_vals = ret_sym.to_numpy(dtype='float64', na_value=np.nan)
                rb_vals = ret_btc.to_numpy(dtype='float64', na_value=np.nan)
                beta_arr = np.full(len(rs_vals), np.nan, dtype=float)
                for i in range(window - 1, len(rs_vals)):
                    rs_win = rs_vals[i - window + 1:i + 1]
                    rb_win = rb_vals[i - window + 1:i + 1]
                    if not (np.isfinite(rs_win).all() and np.isfinite(rb_win).all()):
                        continue
                    var_b = np.var(rb_win, ddof=1)
                    if var_b <= 0 or not np.isfinite(var_b):
                        continue
                    cov = np.cov(rs_win, rb_win, ddof=1)[0, 1]
                    if not np.isfinite(cov):
                        continue
                    beta_arr[i] = cov / var_b
                beta_series = pd.Series(beta_arr, index=ret_sym.index)
                out['beta_btc_20p'] = beta_series
            if eth is not None:
                eth_close = _align_close(eth)
                if eth_close is not None and btc_close is not None:
                    out['eth_btc_ratio'] = (eth_close / btc_close)

        # Shift any price-derived direct features to enforce PIT
        shift_cols = [
            'return_1p', 'volatility_20p', 'momentum_10p', 'rsi_14', 'close_vs_sma50',
            'round_proximity', 'btc_corr_20p', 'beta_btc_20p', 'eth_btc_ratio'
        ]
        for c in shift_cols:
            if c in out.columns:
                out[c] = out[c].shift(1)

        core_cols = ['open','high','low','close','volume']
        out = out.dropna(subset=[c for c in core_cols if c in out.columns])
        out = out.set_index('timestamp')
        return out

from pathlib import Path
import pandas as pd
from typing import Optional, List  # added List

class Lakehouse:
    """
    v4.7.5 layout:
      data/{exchange}/{timeframe}/{YYYY}/{MM}/{CANONICAL}.parquet
    """
    def __init__(self, data_dir: str = "./data", preferred_exchange: Optional[str] = None):
        self.base_path = Path(data_dir)
        self.preferred_exchange = preferred_exchange or "coinbase"

    def _path(self, canonical_symbol: str, timeframe: str, exchange: Optional[str] = None) -> Path:
        exchange_to_use = exchange or self.preferred_exchange
        return self.base_path / exchange_to_use / timeframe

    def get_data(self, canonical_symbol: str, timeframe: str, exchange: Optional[str] = None, limit_months: Optional[int] = None) -> pd.DataFrame:
        p = self._path(canonical_symbol, timeframe, exchange)
        # Explicit year/month two-level traversal (portable; avoids platform-specific ** quirks)
        files = sorted(p.glob(f"[0-9][0-9][0-9][0-9]/[0-9][0-9]/{canonical_symbol}.parquet"))
        if limit_months and limit_months > 0:
            files = files[-limit_months:]
        if not files:
            # fallback to legacy layout
            legacy_exchange = exchange or self.preferred_exchange
            legacy = (self.base_path / legacy_exchange / timeframe / f"{canonical_symbol}.parquet")
            if legacy.exists():
                return pd.read_parquet(legacy)
            # even older legacy
            legacy2 = (self.base_path / timeframe / f"{canonical_symbol}.parquet")
            return pd.read_parquet(legacy2) if legacy2.exists() else pd.DataFrame()
        parts = [pd.read_parquet(f) for f in files]
        if not parts: return pd.DataFrame()
        df = pd.concat(parts).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        if df.empty:
            return df
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        # PIT safety: drop possibly open last bar
        if len(df) > 1:
            tf_str = timeframe.lower()
            dur_s = 0
            if tf_str.endswith('m') and tf_str[:-1].isdigit():
                dur_s = int(tf_str[:-1]) * 60
            elif tf_str.endswith('h') and tf_str[:-1].isdigit():
                dur_s = int(tf_str[:-1]) * 3600
            elif tf_str.endswith('d') and tf_str[:-1].isdigit():
                dur_s = int(tf_str[:-1]) * 86400
            if dur_s > 0:
                import time as _t
                last_ts = int(df.index[-1].timestamp())
                if int(_t.time()) - last_ts < dur_s:
                    df = df.iloc[:-1]
        return df.reset_index()

    def get_available_symbols(self, timeframe: str, exchange: Optional[str] = None) -> list[str]:
        """Return a deduplicated, sorted list of canonical symbols with data for timeframe.

        PIT Note: Only inspects file presence (closed bars already materialized). No
        loading of partial/in-flight parquet content occurs here. Duplicates can
        arise because each month stores a separate parquet for the same symbol;
        this method collapses them into unique canonical identifiers.
        """
        exchange_to_use = exchange or self.preferred_exchange
        p = self.base_path / exchange_to_use / timeframe
        if not p.exists():
            # legacy fallback (flat layout) — keep deterministic ordering
            p2 = self.base_path / timeframe
            return sorted({q.stem for q in p2.glob("*.parquet")}) if p2.exists() else []
        # Collect stems across year/month partitions then dedupe
        symbols: set[str] = {q.stem for q in p.glob("**/*.parquet")}
        return sorted(symbols)

import json
from pathlib import Path
import pandas as pd
from typing import Dict, Any
import time as _time

REQUIRED_COLS = ["timestamp", "open", "high", "low", "close", "volume"]
# Exposed timeframe mapping (ms) to satisfy tests and external callers needing consistency with harvester.
TF_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "6h": 21_600_000,  # added for alias / aggregation consistency
    "1d": 86_400_000,
}

def validate_frame(df: pd.DataFrame, timeframe_ms: int, *, max_gap_pct: float | None = None) -> Dict[str, Any]:
    """Validate OHLCV frame; enforce schema, UTC, numeric fields, gaps & dupes.

    PIT: No forward synthesis; fail-closed on schema anomalies.
    If max_gap_pct provided and gap ratio <= threshold, mark as non-degraded while keeping gap count.
    """
    degraded = False; gaps = 0; dupes = 0; msgs: list[str] = []
    reason = 'ok'
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        msgs.append(f"missing required columns: {missing}")
        return {"degraded": True, "gaps": 0, "dupes": 0, "messages": msgs, "df": df, "gap_ratio": 1.0}
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        except Exception:
            msgs.append("timestamp not datetime64")
            return {"degraded": True, "gaps": 0, "dupes": 0, "messages": msgs, "df": df, "gap_ratio": 1.0}
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    # numeric coercion
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["timestamp","open","high","low","close","volume"])
    if len(df) < before:
        degraded = True; msgs.append(f"dropped {before-len(df)} non-numeric rows")
    df = df.sort_values("timestamp")
    before = len(df)
    df = df.drop_duplicates(subset=["timestamp"]); dupes = before - len(df)
    # Drop any open (incomplete) bar at the tail (timestamp >= now)
    # Determine if last bar is beyond the last fully closed bar boundary
    cur_ms = int(pd.Timestamp.now(tz='UTC').value // 1_000_000)
    if not df.empty:
        last_ms = int(pd.to_datetime(df["timestamp"].iloc[-1]).value // 1_000_000)
        cur_bucket_start = (cur_ms // int(timeframe_ms)) * int(timeframe_ms)
        # Consider bar open if it belongs to the current in-progress bucket (start > bucket_start)
        if last_ms > cur_bucket_start:
            df = df.iloc[:-1]
            degraded = True
            msgs.append("dropped open tail bar")
            reason = 'open_bar_removed'

    gap_ratio = 0.0
    if len(df) > 3 and reason == 'ok':
        # Estimate expected bars from first/last timestamps and tf step
        ts0 = int(df['timestamp'].iloc[0].value // 1_000_000)
        tsN = int(df['timestamp'].iloc[-1].value // 1_000_000)
        expected = max(1, (tsN - ts0) // int(timeframe_ms) + 1)
        missing = max(0, expected - len(df))
        if missing > 0:
            degraded = True
            gap_ratio = missing / expected
            if max_gap_pct is not None:
                if gap_ratio <= max_gap_pct:
                    reason = 'has_gaps'
                else:
                    reason = 'gap_ratio_exceeded'
    accepted = reason != 'gap_ratio_exceeded'
    return {"accepted": accepted, "degraded": degraded, "gaps": gaps, "dupes": dupes, "messages": msgs, "df": df, "gap_ratio": gap_ratio, "reason": reason}

def atomic_write_parquet(df: pd.DataFrame, out_path: Path, quality_meta: Dict[str, Any]) -> None:
    """Atomically write parquet + sidecar quality JSON (fail‑closed semantics).

    PIT Note: Writes only validated, historical data; caller must ensure no future bars.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)
    qpath = out_path.with_suffix(".quality.json")
    meta = {k: v for k, v in quality_meta.items() if k != "df"}
    qpath.write_text(json.dumps(meta, indent=2))

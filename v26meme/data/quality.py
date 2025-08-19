import json
from pathlib import Path
import pandas as pd
from typing import Dict, Any

REQUIRED_COLS = ["timestamp", "open", "high", "low", "close", "volume"]
# Exposed timeframe mapping (ms) to satisfy tests and external callers needing consistency with harvester.
TF_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

def validate_frame(df: pd.DataFrame, timeframe_ms: int, *, max_gap_pct: float | None = None) -> Dict[str, Any]:
    """Validate OHLCV frame; enforce schema, UTC, numeric fields, gaps & dupes.

    PIT: No forward synthesis; fail-closed on schema anomalies.
    If max_gap_pct provided and gap ratio <= threshold, mark as non-degraded while keeping gap count.
    """
    degraded = False; gaps = 0; dupes = 0; msgs: list[str] = []
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
    gap_ratio = 0.0
    if len(df) > 3:
        diffs = df["timestamp"].astype("int64").diff().dropna() // 10**6
        exp = int(timeframe_ms); gap_thresh = int(exp * 1.5)
        gaps = int((diffs > gap_thresh).sum())
        if gaps > 0:
            degraded = True
            expected_intervals = max(1, len(df) + gaps - 1)
            gap_ratio = gaps / expected_intervals
            if max_gap_pct is not None and gap_ratio <= max_gap_pct:
                # Accept despite gaps; downgrade degraded flag while retaining message
                msgs.append(f"accepting with gaps={gaps} gap_ratio={gap_ratio:.3f} <= threshold {max_gap_pct:.3f}")
                degraded = False
    return {"degraded": degraded, "gaps": gaps, "dupes": dupes, "messages": msgs, "df": df, "gap_ratio": gap_ratio}

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

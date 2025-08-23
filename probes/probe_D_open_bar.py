"""Probe D: Open / In-Progress Bar Leakage & Alignment Audit

Objective:
  Detect if harvester writes the currently open (still forming) bar for any timeframe,
  and validate timestamp alignment to expected interval grid.

Heuristics (read-only, deterministic snapshot):
  - For each (tf) in scope, load symbol parquet tail, get last bar timestamp (UTC).
  - Compute duration_s for timeframe.
  - Now = current epoch (UTC). Age = now - last_bar_start.
  - If age < duration_s, last bar is still open -> open_bar_leak = true.
  - Alignment check: all timestamps modulo duration_s should be 0 (within entire loaded tail subset).
    We sample only the last max_check_rows (default 5000) for speed.
  - Report counts of misaligned timestamps (should be 0) and whether last bar appears open.

Outputs JSON object:
  {
    timeframe: {
       'rows_checked': int,
       'duration_s': int,
       'last_bar': 'ISO8601',
       'last_bar_age_s': int,
       'open_bar_flag': bool,
       'misaligned_timestamps': int,
       'misalignment_examples': [... up to 5 ...]
    }, ...
  }

No mutations; safe to run multiple times. If open_bar_flag true for any tf, aggregation / feature PIT safety is violated.
"""
from __future__ import annotations
import pathlib, time, json, pandas as pd
from typing import Dict, List

EX = 'coinbase'
SYM = 'BTC_USD_SPOT'
TIMEFRAMES = ['1m','5m','15m','1h']  # extend if needed
BASE = pathlib.Path('data') / EX
MAX_CHECK_ROWS = 5000


def tf_duration_seconds(tf: str) -> int:
    tf = tf.lower()
    if tf.endswith('m') and tf[:-1].isdigit():
        return int(tf[:-1]) * 60
    if tf.endswith('h') and tf[:-1].isdigit():
        return int(tf[:-1]) * 3600
    if tf.endswith('d') and tf[:-1].isdigit():
        return int(tf[:-1]) * 86400
    raise ValueError(f"Unsupported timeframe {tf}")


def list_symbol_files(tf: str, sym: str):
    return sorted((BASE / tf).glob(f"*/??/{sym}.parquet"))


def load_tail(tf: str, sym: str) -> pd.DataFrame:
    files = list_symbol_files(tf, sym)
    if not files:
        return pd.DataFrame()
    parts: List[pd.DataFrame] = []
    for p in files:
        try:
            parts.append(pd.read_parquet(p))
        except Exception as e:
            print(json.dumps({'warn_read': str(p), 'error': str(e)}))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    if 'timestamp' not in df.columns:
        return pd.DataFrame()
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
    if len(df) > MAX_CHECK_ROWS:
        df = df.iloc[-MAX_CHECK_ROWS:]
    return df.reset_index(drop=True)


def analyze_tf(tf: str) -> Dict[str, any]:
    out: Dict[str, any] = {}
    df = load_tail(tf, SYM)
    if df.empty:
        out.update({'rows_checked': 0, 'status': 'no_data'})
        return out
    dur = tf_duration_seconds(tf)
    last_ts = df.timestamp.iloc[-1]
    last_epoch = int(last_ts.timestamp())
    now_epoch = int(time.time())
    age = now_epoch - last_epoch
    open_flag = age < dur
    # alignment check on sample
    ts_seconds = df.timestamp.view('int64') // 1_000_000_000  # convert to seconds
    misaligned_mask = (ts_seconds % dur) != 0
    misaligned_count = int(misaligned_mask.sum())
    examples: List[str] = []
    if misaligned_count:
        examples = [df.timestamp.iloc[i].isoformat() for i in list(df.index[misaligned_mask])[:5]]
    out.update({
        'rows_checked': int(len(df)),
        'duration_s': dur,
        'last_bar': last_ts.isoformat(),
        'last_bar_age_s': age,
        'open_bar_flag': bool(open_flag),
        'misaligned_timestamps': misaligned_count,
        'misalignment_examples': examples,
    })
    return out


def main():
    report: Dict[str, any] = {}
    for tf in TIMEFRAMES:
        report[tf] = analyze_tf(tf)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()

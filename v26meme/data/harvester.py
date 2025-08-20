# Event-sourced, lane-aware, resumable, QA-gated harvester
import yaml, time, ccxt  # ensure ccxt imported for RateLimitExceeded reference
from pathlib import Path
from loguru import logger
import pandas as pd
from datetime import datetime, timezone, timedelta
import json, ast, math
from typing import List, Dict, Any, Iterable, Set, Awaitable, Union, Optional, Tuple
import glob  # NEW: for aggregation file discovery
from collections.abc import Awaitable as _Awaitable  # ensure available early

def _sync(val):  # always defined before any use
    try:
        if isinstance(val, _Awaitable):
            return None
    except Exception:
        return val
    return val

from v26meme.core.state import StateManager
from v26meme.data.checkpoints import Checkpoints
from v26meme.data.token_bucket import TokenBucket
from v26meme.data.quality import validate_frame, atomic_write_parquet
from v26meme.registry.canonical import make_canonical, venue_symbol_for

TF_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "6h": 21_600_000,  # added for coinbase aliasing
    "1d": 86_400_000,
}

def _write_partitioned_parquet(base_dir: Path, exchange: str, tf_resolved: str, canonical: str, df: pd.DataFrame, meta: Dict[str, Any]) -> int:
    """Partition-aware atomic write.

    Splits df by UTC year/month and writes each partition independently to
    data/<exchange>/<tf>/<YYYY>/<MM>/<canonical>.parquet
    preserving PIT correctness by routing rows via their intrinsic timestamps.
    Returns max timestamp (ms) for checkpoint.
    """
    if df.empty:
        return 0
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    max_ts_ms = int(df['timestamp'].max().timestamp()*1000)
    for (yy, mm), part in df.groupby([df['timestamp'].dt.year, df['timestamp'].dt.month]):
        out_path = base_dir / exchange / tf_resolved / f"{yy:04d}" / f"{mm:02d}" / f"{canonical}.parquet"
        atomic_write_parquet(part.sort_values('timestamp'), out_path, {
            **meta,
            'partition_year': int(yy),
            'partition_month': int(mm),
            'rows_partition': len(part),
            'last_ts_partition': int(part['timestamp'].max().timestamp()*1000)
        })
    return max_ts_ms

# NEW helper: optional sparse flat-fill (zero-volume synthetic bars) for moderate gaps
def _flat_fill_sparse(df: pd.DataFrame, tf_ms: int, max_fill_bars: int = 120) -> pd.DataFrame:
    try:
        if df.empty: return df
        df = df.sort_values('timestamp').drop_duplicates('timestamp')
        start, end = df['timestamp'].min(), df['timestamp'].max()
        freq = pd.Timedelta(milliseconds=tf_ms)
        idx = pd.date_range(start, end, freq=freq, tz='UTC', inclusive='both')
        base = df.set_index('timestamp').reindex(idx)
        # Count how many new bars would be synthesized
        synth_count = base['close'].isna().sum()
        if synth_count > max_fill_bars:
            return df  # abort; too sparse
        for col in ['open','high','low','close']:
            base[col] = base[col].ffill()
        base['volume'] = base['volume'].fillna(0.0)
        out = base.reset_index().rename(columns={'index':'timestamp'})
        out['synthetic_zero_volume'] = out['volume'] == 0.0
        return out
    except Exception:
        return df

def _cfg(path="configs/config.yaml"):
    with open(path,"r") as f: return yaml.safe_load(f)

def _exchange(ex_id: str):
    ex = getattr(ccxt, ex_id)()
    # Harden network behavior: rely on ccxt internal limiter + sane timeout
    ex.enableRateLimit = True
    try:
        if not getattr(ex, 'timeout', None) or ex.timeout < 10_000:
            ex.timeout = 10_000  # 10s default
    except Exception:
        pass
    ex.load_markets()
    return ex

def _resolve_timeframe_for_exchange(ex_id: str, ex, tf: str, cfg: dict) -> str | None:
    """Return a venue-supported timeframe.

    Order:
      1) Direct support in exchange.timeframes
      2) Alias via registry.timeframe_aliases_by_venue[ex_id][tf]
      3) None (skip) if not supported.
    If the exchange lacks a .timeframes map, keep original (optimistic).
    """
    supported = set((getattr(ex, "timeframes", {}) or {}).keys())
    if supported:
        if tf in supported:
            return tf
        alias = ((cfg.get("registry") or {}).get("timeframe_aliases_by_venue") or {}) \
            .get(ex_id, {}) \
            .get(tf)
        if alias and alias in supported:
            return alias
        return None
    return tf

SUPPRESS_KEY_PREFIX = "harvest:suppress"  # hash key per cycle to summarize
MISS_HASH = "harvest:unresolvable:attempts"

def _safe_int(val, default: int = -1) -> int:
    try:
        if hasattr(val, '__await__'):
            return default  # avoid awaiting in sync context
        if isinstance(val, bytes):
            try: return int(val.decode('utf-8'))
            except Exception: return default
        return int(val)
    except Exception:
        return default

def _is_suppressed(state: StateManager, exchange: str, canonical: str, ttl_minutes: int) -> bool:
    k = f"harvest:suppress:{exchange}:{canonical}"
    try:
        ttl_raw = state.r.ttl(k)
        ttl_val = _safe_int(ttl_raw, -2)
    except Exception:
        ttl_val = -2
    # Redis TTL semantics: -2 key missing, -1 no expiry (treat as still suppressed), >0 seconds remaining
    return ttl_val == -1 or ttl_val > 0

def _mark_suppressed(state: StateManager, exchange: str, canonical: str, ttl_minutes: int):
    k = f"harvest:suppress:{exchange}:{canonical}"
    try:
        state.r.setex(k, ttl_minutes*60, 1)
    except Exception:
        pass

def _build_plan(cfg, state: StateManager):
    harv = cfg["harvester"]
    plan = {}
    core = set(harv.get("core_symbols", []))
    dyn = set()
    if harv.get("dynamic_enabled", True):
        latest = state.get("data:screener:latest:canonicals") or []
        dyn.update(latest)
        moon = state.get("lane:moonshot:gainers") or []
        dyn.update([f"{b}_USD_SPOT" for b in moon])
        # NOTE: Removed direct EIL queue scan here; queue is now ONLY drained once via _enrich_plan_with_queue
    canonicals = sorted(core.union(dyn)) if dyn else sorted(core)

    tfs_core = harv["timeframes_by_lane"]["core"]
    tfs_moon = harv["timeframes_by_lane"]["moonshot"]

    miss_threshold = int(((harv.get("availability") or {}).get("miss_threshold") or 3))
    suppress_ttl = int(((harv.get("availability") or {}).get("suppress_ttl_minutes") or 360))

    # Pre-fetch exchange objects for availability probing (lightweight symbol mapping build)
    # We'll probe only once per (exchange, canonical) this cycle.
    exchanges: Dict[str, Any] = {}
    for ex_id in cfg["data_source"]["exchanges"]:
        try:
            exchanges[ex_id] = _exchange(ex_id)
        except Exception:
            continue

    for c in canonicals:
        lane_tfs = tfs_moon if c.startswith(tuple([b for b in (state.get("lane:moonshot:gainers") or [])])) else tfs_core
        for tf in lane_tfs:
            if tf not in TF_MS: continue
            plan.setdefault(tf, set()).add(c)

    # Availability filter: remove canonicals suppressed or exceeding miss threshold for each exchange
    filtered_plan: Dict[str, set[str]] = {}
    for tf, cset in plan.items():
        keep: set[str] = set()
        for canon in cset:
            drop_all = True
            for ex_id, ex in exchanges.items():
                if _is_suppressed(state, ex_id, canon, suppress_ttl):
                    continue
                try:
                    sym = venue_symbol_for(ex, canon)
                except Exception:
                    sym = None
                if sym:
                    drop_all = False
                    # success -> clear miss counter so symbol can recover after prior failures
                    try: state.r.hdel(MISS_HASH, f"{ex_id}:{canon}")
                    except Exception: pass
                else:
                    try:
                        state.r.hincrby(MISS_HASH, f"{ex_id}:{canon}", 1)
                        raw_val = state.r.hget(MISS_HASH, f"{ex_id}:{canon}")
                        misses = _safe_int(raw_val, 0)
                    except Exception:
                        misses = 0
                    if misses >= miss_threshold:
                        _mark_suppressed(state, ex_id, canon, suppress_ttl)
                        # reset counter upon suppression to avoid immediate re-suppress after TTL
                        try: state.r.hdel(MISS_HASH, f"{ex_id}:{canon}")
                        except Exception: pass
            if not drop_all:
                keep.add(canon)
        if keep:
            filtered_plan[tf] = keep
    if filtered_plan != plan:
        removed = {tf: list(set(plan[tf]) - set(filtered_plan.get(tf, set()))) for tf in plan.keys()}
        logger.info(f"[harvest] availability_filter removed={removed}")
    return filtered_plan  # {tf: {canonical,...}}

def _out_path(base_dir: Path, exchange: str, timeframe: str, canonical: str) -> Path:
    now = datetime.now(timezone.utc)
    return base_dir / exchange / timeframe / f"{now.year:04d}" / f"{now.month:02d}" / f"{canonical}.parquet"

def _normalize_ohlcv_rows(ohlcv: List[List[float]]) -> List[Dict[str, Any]]:
    """Defensive normalizer for raw OHLCV rows.
    Skips malformed rows; coerces numeric fields; default volume=0 when missing.
    """
    out: List[Dict[str, Any]] = []
    for t in (ohlcv or []):
        if not isinstance(t, (list, tuple)) or len(t) < 5:
            continue
        try:
            ts, o, h, l, c = t[0], t[1], t[2], t[3], t[4]
            v = t[5] if len(t) > 5 and t[5] is not None else 0.0
            row = {
                "timestamp": pd.to_datetime(int(ts), unit="ms", utc=True),
                "open": float(o), "high": float(h), "low": float(l), "close": float(c),
                "volume": float(v)
            }
            out.append(row)
        except Exception:
            continue
    return out

def _safe_parse_eil(item: str | bytes | dict) -> Dict[str, Any] | None:
    """Safe parse for EIL queue entries (JSON or literal dict)."""
    if isinstance(item, dict):
        return item
    if isinstance(item, bytes):
        try:
            item = item.decode('utf-8')
        except Exception:
            return None
    if not isinstance(item, str):
        return None
    s = item.strip()
    if not s:
        return None
    try:
        import json as _json
        obj = _json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    try:
        import ast as _ast
        obj = _ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None

def _identify_gaps(df: pd.DataFrame, tf_ms: int) -> List[tuple[int,int]]:
    gaps: List[tuple[int,int]] = []
    ts = (df['timestamp'].astype('int64') // 10**6).tolist()
    if len(ts) < 2: return gaps
    thresh = int(tf_ms * 1.5)
    for a, b in zip(ts, ts[1:]):
        if b - a > thresh:
            gaps.append((a, b))  # gap between (a, b)
    return gaps

def _fetch_window(exchange, market: str, tf: str, since_ms: int, limit: int, bucket: TokenBucket) -> List[List[float]]:
    bucket.consume(1)
    return exchange.fetch_ohlcv(market, tf, since=since_ms, limit=limit)

def _repair_gaps(exchange, market: str, tf: str, tf_ms: int, df: pd.DataFrame, bucket: TokenBucket, max_attempts: int = 3) -> List[Dict[str, Any]]:
    """Attempt focused backfills for detected gaps; returns additional normalized rows."""
    added: List[Dict[str, Any]] = []
    gaps = _identify_gaps(df, tf_ms)
    if not gaps:
        return added
    attempts = 0
    for start_ms, end_ms in gaps:
        if attempts >= max_attempts: break
        miss_count = int((end_ms - start_ms) // tf_ms) - 1
        if miss_count <= 0: continue
        # Fetch starting slightly before gap start
        since = start_ms - tf_ms * 2
        try:
            ohlcv = _fetch_window(exchange, market, tf, since, min(1000, miss_count + 10), bucket)
        except Exception:
            continue
        rows = _normalize_ohlcv_rows(ohlcv)
        if not rows: continue
        # Keep rows that fill timestamps strictly inside (start_ms, end_ms)
        for r in rows:
            tms = int(r['timestamp'].timestamp() * 1000)
            if start_ms + tf_ms <= tms <= end_ms - tf_ms:
                added.append(r)
        attempts += 1
    return added

def _drain_eil_queue(state: StateManager, limit: int = 200) -> set[str]:
    """Incrementally drain up to 'limit' on-demand harvest requests without dropping overflow.

    Preserves remaining queue entries for subsequent cycles (no delete of key).
    """
    canonicals: set[str] = set()
    count = 0
    try:
        while count < limit:
            raw = state.r.lpop("eil:harvest:requests")
            raw = _sync(raw)
            if raw is None:
                break
            if isinstance(raw, (bytes, str, dict)):
                d = _safe_parse_eil(raw)  # safe types only
                if d and isinstance(d.get("canonical"), str) and d["canonical"]:
                    canonicals.add(d["canonical"])
            count += 1
        if canonicals:
            logger.info(f"Drained {len(canonicals)} canonicals from eil:harvest:requests (remaining entries preserved)")
    except Exception as e:
        logger.debug(f"Queue drain error: {e}")
    return canonicals

def _enrich_plan_with_queue(plan: Dict[str, set[str]], state: StateManager) -> Dict[str, set[str]]:
    """Add any queued on-demand canonicals into every timeframe's plan."""
    queued = _drain_eil_queue(state)
    if queued:
        for tf in list(plan.keys()):
            plan[tf] = plan[tf].union(queued)
        logger.info(f"Added {len(queued)} queued canonicals to harvest plan")
    return plan

def _maybe_aggregate_timeframes(cfg: dict, state: 'StateManager', base_dir: Path) -> None:
    """Optionally build synthetic higher timeframes from a lower source timeframe.

    PIT / Determinism:
      - Uses only already-ingested, closed lower-TF bars (source parquet files on disk).
      - Excludes the current in-flight bar by cutting at (now - target_tf_ms).
    Guardrails:
      - Only aggregates symbols whose native higher-TF coverage is shallow (< min_native_rows_threshold actual rows) to avoid redundant work.
      - Writes via atomic_write_parquet to maintain schema & metadata consistency.
      - Never overwrites native venue data if present; instead union + de-duplicate by timestamp.
    Config:
      harvester.aggregate_timeframes.enabled: bool
      harvester.aggregate_timeframes.from: source timeframe (e.g. '5m')
      harvester.aggregate_timeframes.to: list of target timeframes (currently we support ['1h'])
      harvester.aggregate_timeframes.min_native_rows_threshold: int (skip if native >= threshold)
      harvester.aggregate_timeframes.max_tail_hours: limit aggregation window (performance bound)
    """
    agg_cfg = (cfg.get('harvester') or {}).get('aggregate_timeframes') or {}
    if not agg_cfg.get('enabled', False):
        return
    src_tf = agg_cfg.get('from')
    targets: List[str] = list(agg_cfg.get('to') or [])
    if not src_tf or '1h' not in targets:
        return  # currently only implement 5m->1h path
    src_ms = TF_MS.get(src_tf)
    if src_ms is None:
        return
    target_tf = '1h'
    target_ms = TF_MS.get(target_tf)
    if target_ms is None:
        return
    min_rows = int(agg_cfg.get('min_native_rows_threshold', 50))
    max_tail_hours = int(agg_cfg.get('max_tail_hours', 72))
    cutoff_ts = datetime.now(timezone.utc) - timedelta(milliseconds=target_ms)

    # Identify low-coverage symbols for target_tf from coverage hash (symbol-level)
    low_cov: Dict[str, set[str]] = {ex: set() for ex in cfg.get('data_source', {}).get('exchanges', [])}
    try:
        cov_keys = state.r.hkeys('harvest:coverage') or []
        cov_keys = _sync(cov_keys) or []  # defensive
        if not isinstance(cov_keys, (list, tuple)):
            cov_iter = []
        else:
            cov_iter = list(cov_keys)
        for raw_key in cov_iter:
            try:
                raw_key = _sync(raw_key)
                if raw_key is None:
                    continue
                if isinstance(raw_key, bytes):
                    key_str = raw_key.decode('utf-8')
                else:
                    key_str = str(raw_key)
                parts = key_str.split(':', 2)
                if len(parts) != 3:
                    continue
                ex_id, tf, canonical = parts
                if tf != target_tf:
                    continue
                data_raw = state.r.hget('harvest:coverage', key_str)
                data_raw = _sync(data_raw)
                if not data_raw:
                    continue
                if isinstance(data_raw, bytes):
                    data_raw = data_raw.decode('utf-8')
                if isinstance(data_raw, str):
                    meta = json.loads(data_raw)
                else:
                    continue
                actual = int(meta.get('actual') or 0)
                if actual < min_rows:
                    low_cov.setdefault(ex_id, set()).add(canonical)
            except Exception:
                continue
    except Exception:
        pass

    aggregated = 0
    improved = 0
    for ex_id, symbols in low_cov.items():
        if not symbols:
            continue
        for canonical in sorted(symbols):
            now = datetime.now(timezone.utc)
            # Load current and previous month to avoid early-month truncation
            months: list[tuple[int,int]] = [(now.year, now.month)]
            prev_month_anchor = (now.replace(day=1) - timedelta(days=1))
            prev_tuple = (prev_month_anchor.year, prev_month_anchor.month)
            if prev_tuple not in months:
                months.append(prev_tuple)
            parts = []
            for yy, mm in months:
                src_path = base_dir / ex_id / src_tf / f"{yy:04d}" / f"{mm:02d}" / f"{canonical}.parquet"
                if src_path.exists():
                    try:
                        parts.append(pd.read_parquet(src_path))
                    except Exception:
                        continue
            if not parts:
                continue
            try:
                df_src = pd.concat(parts, ignore_index=True).drop_duplicates(subset=['timestamp'])
            except Exception:
                continue
            if df_src.empty:
                continue
            # Tail window restriction
            tail_start = cutoff_ts - timedelta(hours=max_tail_hours)
            df_src = df_src[df_src['timestamp'] >= tail_start]
            if df_src.empty:
                continue
            # Ensure timestamp dtype
            if not pd.api.types.is_datetime64_any_dtype(df_src['timestamp']):
                try:
                    df_src['timestamp'] = pd.to_datetime(df_src['timestamp'], utc=True)
                except Exception:
                    continue
            df_src = df_src.sort_values('timestamp')
            # Build aggregated 1h bars using closed intervals only (exclude bar with end > cutoff)
            df_src = df_src[df_src['timestamp'] <= cutoff_ts]
            if df_src.empty:
                continue
            df_src = df_src.set_index('timestamp')
            try:
                ohlc = df_src.resample('1H', label='right', closed='right').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                })
            except Exception:
                continue
            ohlc = ohlc.dropna(subset=['open','high','low','close'])
            if ohlc.empty:
                continue
            ohlc = ohlc.reset_index()
            out_path = base_dir / ex_id / target_tf / f"{now.year:04d}" / f"{now.month:02d}" / f"{canonical}.parquet"
            # Merge with existing if present
            if out_path.exists():
                try:
                    existing = pd.read_parquet(out_path)
                    before_rows = len(existing)
                    merged = pd.concat([existing, ohlc]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                    if len(merged) > before_rows:
                        improved += 1
                    final_df = merged
                except Exception:
                    final_df = ohlc
            else:
                improved += 1
                final_df = ohlc
            try:
                atomic_write_parquet(final_df, out_path, {
                    'aggregated_from': src_tf,
                    'rows': len(final_df),
                    'last_ts': int(final_df['timestamp'].max().timestamp()*1000),
                    'source_tail_hours': max_tail_hours,
                    'native_rows_threshold': min_rows,
                })
                aggregated += 1
            except Exception:
                continue
    if aggregated:
        logger.info(f"[harvest] aggregated_timeframes built target=1h from={src_tf} symbols={aggregated} improved={improved}")

def run_once(cfg, state: StateManager, partial_mode: bool | None = None):
    """Harvest one cycle of OHLCV data (event-sourced, QA-gated).

    Parameters
    ----------
    cfg : dict
        Full system configuration (contains harvester.* keys).
    state : StateManager
        Shared state / Redis wrapper (checkpoints, coverage hashes, etc.).
    partial_mode : bool | None
        When True (and harvester.partial_harvest enabled) process ONLY a single
        timeframe per invocation (rotating across timeframes each call). This
        reduces cold‑start wall time before the research loop can begin using
        the highest‑priority data. When None, derives from
        cfg['harvester']['partial_harvest'] (default False).

    PIT Note:
        Partial mode still uses the same deterministic ordering (priority
        symbols first, then remaining alphabetically) and writes identical
        parquet output over multiple invocations as the original monolithic
        run. No lookahead or reordering of bars occurs.
    """
    if partial_mode is None:
        partial_mode = bool((cfg.get('harvester') or {}).get('partial_harvest', False))
    logger.info("[harvest] run_once invoked (event-sourced cycle start)" + (" partial" if partial_mode else ""))
    base_dir = Path("./data")
    quotas = cfg["harvester"].get("quotas", {})
    plan = _build_plan(cfg, state)
    plan = _enrich_plan_with_queue(plan, state)
    ckp = Checkpoints(state)

    # PRIORITY symbol ordering — keep configured priority symbols first
    priority_list = list((cfg.get('harvester') or {}).get('priority_symbols') or [])
    rank = {sym: i for i, sym in enumerate(priority_list)}
    def _priority_sort(iterable):
        return sorted(iterable, key=lambda s: (rank.get(s, 10_000), s))

    # PARTIAL timeframe rotation (single timeframe per invocation if enabled)
    if partial_mode and plan:
        ordered_tfs = [tf for tf in ["1m","5m","15m","1h","4h","1d"] if tf in plan]
        if ordered_tfs:
            idx_key = 'harvest:partial:tf_index'
            cur_idx = int(state.get(idx_key) or 0)
            tf_chosen = ordered_tfs[cur_idx % len(ordered_tfs)]
            # prune plan to only chosen timeframe
            plan = {tf_chosen: plan[tf_chosen]}
            state.set(idx_key, (cur_idx + 1) % len(ordered_tfs))
            logger.info(f"[harvest] partial_mode timeframe={tf_chosen} (idx={cur_idx})")

    max_gap_pct_accept = float(cfg["harvester"].get("max_gap_pct_accept", 0.10))
    bootstrap_overrides = cfg["harvester"].get("bootstrap_days_override", {"1m":2, "5m":7})
    staged_cfg = (cfg["harvester"].get("staged_backfill") or {})
    staged_enabled = bool(staged_cfg.get("enabled", True))
    targets_days: Dict[str, List[int]] = staged_cfg.get("targets_days", {})
    buckets: Dict[str, TokenBucket] = {}
    exchanges = {}
    for ex_id in cfg["data_source"]["exchanges"]:
        q = quotas.get(ex_id, {"max_requests_per_min": 30, "min_sleep_ms": 200})
        buckets[ex_id] = TokenBucket(q["max_requests_per_min"], q["min_sleep_ms"])
        exchanges[ex_id] = _exchange(ex_id)
    accepted_stats: Dict[str, Dict[str,int]] = {}

    for ex_id, ex in exchanges.items():
        for tf, canon_set in plan.items():
            tf_resolved = _resolve_timeframe_for_exchange(ex_id, ex, tf, cfg)
            if not tf_resolved:
                logger.debug(f"skip {ex_id} {tf}: unsupported timeframe on venue")
                continue
            if tf_resolved != tf:
                logger.debug(f"{ex_id} timeframe alias {tf} -> {tf_resolved}")
            tf_ms = TF_MS.get(tf_resolved)
            if tf_ms is None:
                logger.debug(f"skip {ex_id} {tf_resolved}: no TF_MS mapping")
                continue
            accepted_stats.setdefault(tf, {"attempt":0, "ok":0})
            stage_days = None
            if staged_enabled and tf in targets_days:
                stage_key = f"harvest:backfill:stage:{tf}"
                stage_idx = int(state.get(stage_key) or 0)
                seq = targets_days.get(tf) or []
                if 0 <= stage_idx < len(seq):
                    stage_days = seq[stage_idx]
            # iterate priority-sorted
            for canonical in _priority_sort(canon_set):
                accepted_stats[tf]["attempt"] += 1
                market = venue_symbol_for(ex, canonical)
                if not market:
                    # fallback attempt: BASE/QUOTE direct symbol if present in markets
                    try:
                        base, quote, _ = canonical.split("_")
                        candidate = f"{base}/{quote}"
                        if candidate in (ex.markets or {}):
                            market = candidate
                    except Exception:
                        market = None
                if not market:
                    continue
                # Checkpoint now keyed by *resolved* timeframe for determinism & alias transparency
                since = ckp.get(ex_id, canonical, tf_resolved)
                if stage_days and tf in ("1m","5m","15m"):
                    target_since = ex.milliseconds() - 86_400_000 * stage_days
                    if since is None or target_since < since - tf_ms * 10:
                        since = target_since
                        logger.debug(f"staged_backfill {ex_id} {tf} {canonical}: target_days={stage_days}")
                if since is None:
                    per_ex_boot = ((cfg.get('harvester') or {}).get('per_exchange_bootstrap') or {})
                    days_candidate = None
                    try:
                        days_candidate = per_ex_boot.get(ex_id, {}).get(tf)
                    except Exception:
                        days_candidate = None
                    base_days = cfg["harvester"]["bootstrap_days_default"].get(tf, 30)
                    override_days = bootstrap_overrides.get(tf, base_days)
                    days = int(days_candidate if days_candidate is not None else override_days)
                    since = ex.milliseconds() - 86_400_000 * days
                limit = 1000
                if ex_id == 'coinbase':
                    limit = 300
                all_rows: List[Dict[str, Any]] = []
                last_ts = since
                try:
                    while True:
                        buckets[ex_id].consume(1)
                        try:
                            ohlcv = ex.fetch_ohlcv(market, tf_resolved, since=last_ts, limit=limit)
                        except ccxt.RateLimitExceeded as rl:  # type: ignore[attr-defined]
                            backoff = max(2.0, (getattr(ex, 'rateLimit', 100) / 1000.0) * 3.0)
                            logger.warning(f"rate_limit backoff {ex_id} tf={tf} sym={canonical} sleep={backoff:.2f}s")
                            try:
                                state.r.hincrby("harvest:rate_limit_hits", ex_id, 1)
                            except Exception:
                                pass
                            time.sleep(backoff)
                            break
                        if not ohlcv: break
                        if last_ts is not None and ohlcv[-1][0] <= (last_ts or 0): break
                        last_ts = ohlcv[-1][0] + tf_ms
                        rows = _normalize_ohlcv_rows(ohlcv)
                        if rows:
                            all_rows.extend(rows)
                        time.sleep(max(0.0, getattr(ex, 'rateLimit', 100)/1000.0))
                        if len(ohlcv) < limit: break
                except Exception as e:
                    logger.opt(exception=True).error(f"{ex_id} {tf} {canonical} harvest error: {e}")
                    state.r.hincrby("harvest:errors", ex_id, 1)
                    continue
                if not all_rows:
                    continue
                df = pd.DataFrame(all_rows)
                expected_rows = None
                if since is not None and last_ts is not None:
                    span_ms = (last_ts - since)
                    expected_rows = max(1, int(math.ceil(span_ms / tf_ms)))
                gap_overrides = ((cfg.get('harvester') or {}).get('gap_accept_overrides') or {}).get(ex_id, {})
                gap_cap = float(gap_overrides.get(tf_resolved, max_gap_pct_accept))
                try:
                    qa = validate_frame(df, tf_ms, max_gap_pct=gap_cap)
                except Exception as e:
                    logger.opt(exception=True).error(f"{ex_id} {tf} {canonical} validation error: {e}")
                    state.r.hincrby("harvest:errors", ex_id, 1)
                    continue
                if qa['degraded']:
                    fb_map = ((cfg.get('harvester') or {}).get('tf_fallback') or {}).get(ex_id, {})
                    fb_tf = fb_map.get(tf_resolved)
                    if fb_tf and fb_tf in TF_MS:
                        logger.info(f"fallback {ex_id} {canonical}: {tf_resolved} too sparse -> {fb_tf}")
                        try:
                            buckets[ex_id].consume(1)
                            fb_ohlcv = ex.fetch_ohlcv(market, fb_tf, since=since, limit=limit)
                            fb_rows = _normalize_ohlcv_rows(fb_ohlcv)
                            if fb_rows:
                                df_fb = pd.DataFrame(fb_rows)
                                gap_cap_fb = float(gap_overrides.get(fb_tf, max_gap_pct_accept))
                                qa2 = validate_frame(df_fb, TF_MS[fb_tf], max_gap_pct=gap_cap_fb)
                                qa = qa2
                                tf_resolved = fb_tf
                                tf_ms = TF_MS[fb_tf]
                        except Exception as e:
                            logger.warning(f"fallback fetch failed {ex_id} {canonical} {fb_tf}: {e}")
                if qa['degraded'] and float(qa.get('gap_ratio',1.0)) <= gap_cap * 1.05:
                    ff = _flat_fill_sparse(qa['df'], tf_ms)
                    if len(ff) > len(qa['df']):
                        try:
                            qa2 = validate_frame(ff, tf_ms, max_gap_pct=gap_cap)
                            if not qa2['degraded']:
                                logger.info(f"flat_fill accepted {ex_id} {canonical} tf={tf_resolved} filled_rows={len(ff)-len(qa['df'])}")
                                qa = qa2
                        except Exception:
                            pass
                if qa.get('gaps',0) > 0 and not qa['degraded']:
                    repair_rows = _repair_gaps(ex, market, tf_resolved, tf_ms, qa['df'], buckets[ex_id])
                    if repair_rows:
                        merged = pd.concat([qa['df'], pd.DataFrame(repair_rows)]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                        try:
                            qa2 = validate_frame(merged, tf_ms, max_gap_pct=gap_cap)
                            qa = qa2
                        except Exception:
                            pass
                elif qa['degraded'] and qa.get('gap_ratio',1.0) <= gap_cap:
                    repair_rows = _repair_gaps(ex, market, tf_resolved, tf_ms, qa['df'], buckets[ex_id])
                    if repair_rows:
                        merged = pd.concat([qa['df'], pd.DataFrame(repair_rows)]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                        try:
                            qa2 = validate_frame(merged, tf_ms, max_gap_pct=gap_cap)
                            qa = qa2
                        except Exception:
                            pass
                actual = len(qa["df"]) if qa.get("df") is not None else 0
                coverage = (actual / expected_rows) if (expected_rows and expected_rows>0) else 0.0
                # Coverage key now uses *resolved* timeframe. Maintain backward write for one release cycle.
                cov_payload = json.dumps({
                    "expected": expected_rows,
                    "actual": actual,
                    "coverage": round(coverage,4),
                    "gaps": qa["gaps"],
                    "gap_ratio": round(float(qa.get("gap_ratio",0.0)),4),
                    "accepted": not qa["degraded"],
                })
                state.hset("harvest:coverage", f"{ex_id}:{tf_resolved}:{canonical}", cov_payload)
                # Backward compatibility write (to be removed after migration window)
                if tf_resolved != tf:
                    state.hset("harvest:coverage:migr_legacy", f"{ex_id}:{tf}:{canonical}", cov_payload)
                if qa["degraded"]:
                    logger.warning(f"Reject {ex_id} {tf_resolved} {canonical}: gaps={qa['gaps']} gap_ratio={qa.get('gap_ratio'):.3f} coverage={coverage:.2f}")
                    continue
                meta = {"rows_added": len(qa["df"]), "gaps_detected": qa["gaps"],
                        "dupes_removed": qa["dupes"], "last_ts": int(qa["df"]['timestamp'].max().timestamp()*1000),
                        "degraded": False, "expected_rows": expected_rows, "coverage": coverage,
                        "gap_ratio": qa.get("gap_ratio", 0.0), "staged_days": stage_days, "tf_resolved": tf_resolved, "gap_cap": gap_cap}
                last_ts_ms = _write_partitioned_parquet(base_dir, ex_id, tf_resolved, canonical, qa["df"], meta)
                if last_ts_ms:
                    ckp.set(ex_id, canonical, tf_resolved, last_ts_ms)
                    state.r.hincrby(f"harvest:bars_written:{ex_id}:{tf_resolved}", "count", len(qa["df"]))
                    state.hset("harvest:checkpoint", f"{ex_id}:{tf_resolved}:{canonical}", last_ts_ms)
                    accepted_stats[tf]["ok"] += 1

    # Stage progression: advance timeframe stage when majority accepted with good gap ratio
    if staged_enabled:
        for tf, stat in accepted_stats.items():
            seq = targets_days.get(tf)
            if not seq or tf not in ("1m","5m","15m"): continue
            stage_key = f"harvest:backfill:stage:{tf}"
            stage_idx = int(state.get(stage_key) or 0)
            if stage_idx >= len(seq)-1: continue
            attempts = stat.get("attempt",0)
            oks = stat.get("ok",0)
            if attempts == 0: continue
            accept_ratio = oks/attempts
            if accept_ratio >= 0.7:  # promotion threshold
                state.set(stage_key, stage_idx+1)
                logger.info(f"staged_backfill advance {tf}: {seq[stage_idx]}d -> {seq[stage_idx+1]}d (accept_ratio={accept_ratio:.2f})")

    # Coverage summary (always logged) — PIT safe (derived from already-ingested cycle stats)
    if accepted_stats:
        summary = {tf: {"attempt": s.get("attempt",0), "ok": s.get("ok",0), "accept_ratio": (s.get("ok",0)/s.get("attempt",1)) if s.get("attempt",0)>0 else 0.0} for tf, s in accepted_stats.items()}
        logger.info(f"[harvest] coverage_summary {summary}")
        state.set("harvest:coverage_summary:last", summary)

    # Synthetic aggregation (after primary coverage summary)
    try:
        _maybe_aggregate_timeframes(cfg, state, base_dir)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"aggregate_timeframes error: {e}")

    if not staged_enabled:
        logger.debug("staged_backfill disabled (accelerated bootstrap mode active)")

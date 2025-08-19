# Event-sourced, lane-aware, resumable, QA-gated harvester
import yaml, time, ccxt
from pathlib import Path
from loguru import logger
import pandas as pd
from datetime import datetime, timezone
import json, ast, math
from typing import List, Dict, Any, Iterable

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

def _cfg(path="configs/config.yaml"):
    with open(path,"r") as f: return yaml.safe_load(f)

def _exchange(ex_id: str):
    ex = getattr(ccxt, ex_id)()
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
        # EIL requests (high priority)
        eil_req = state.r.lrange("eil:harvest:requests", 0, -1) or []
        if not isinstance(eil_req, (list, tuple)):
            eil_list = []
        else:
            eil_list = eil_req
        for item in eil_list:
            try:
                d = _safe_parse_eil(item)
                if not d:
                    continue
                canon = d.get("canonical")
                if isinstance(canon, str) and canon:
                    dyn.add(canon)
            except Exception:
                continue
        if eil_list: state.r.delete("eil:harvest:requests")
    canonicals = sorted(core.union(dyn)) if dyn else sorted(core)

    tfs_core = harv["timeframes_by_lane"]["core"]
    tfs_moon = harv["timeframes_by_lane"]["moonshot"]
    for c in canonicals:
        lane_tfs = tfs_moon if c.startswith(tuple([b for b in (state.get("lane:moonshot:gainers") or [])])) else tfs_core
        for tf in lane_tfs:
            if tf not in TF_MS: continue
            plan.setdefault(tf, set()).add(c)
    return plan  # {tf: {canonical,...}, ...}

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

def run_once(cfg, state: StateManager):
    base_dir = Path("./data")
    quotas = cfg["harvester"].get("quotas", {})
    plan = _build_plan(cfg, state)
    ckp = Checkpoints(state)

    max_gap_pct_accept = float(cfg["harvester"].get("max_gap_pct_accept", 0.10))  # 10% default
    bootstrap_overrides = cfg["harvester"].get("bootstrap_days_override", {"1m":2, "5m":7})

    staged_cfg = (cfg["harvester"].get("staged_backfill") or {})
    staged_enabled = bool(staged_cfg.get("enabled", True))
    targets_days: Dict[str, List[int]] = staged_cfg.get("targets_days", {})

    # token buckets per exchange
    buckets: Dict[str, TokenBucket] = {}
    exchanges = {}
    for ex_id in cfg["data_source"]["exchanges"]:
        q = quotas.get(ex_id, {"max_requests_per_min": 30, "min_sleep_ms": 200})
        buckets[ex_id] = TokenBucket(q["max_requests_per_min"], q["min_sleep_ms"])
        exchanges[ex_id] = _exchange(ex_id)

    # Track per-timeframe acceptance stats for staged progression
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

            # Determine staged target days (intraday only)
            stage_days = None
            if staged_enabled and tf in targets_days:
                stage_key = f"harvest:backfill:stage:{tf}"
                stage_idx = int(state.get(stage_key) or 0)
                seq = targets_days.get(tf) or []
                if 0 <= stage_idx < len(seq):
                    stage_days = seq[stage_idx]

            for canonical in sorted(canon_set):
                accepted_stats[tf]["attempt"] += 1
                market = venue_symbol_for(ex, canonical)
                if not market:
                    logger.debug(f"skip {ex_id} {tf} {canonical}: no resolvable market on venue")
                    continue
                since = ckp.get(ex_id, canonical, tf)
                # Staged backfill override: re-fetch broader window until target depth satisfied (rewrite allowed)
                if stage_days and tf in ("1m","5m","15m"):
                    target_since = ex.milliseconds() - 86_400_000 * stage_days
                    if since is None or target_since < since - tf_ms * 10:  # if our current dataset shorter than target
                        since = target_since
                        logger.debug(f"staged_backfill {ex_id} {tf} {canonical}: target_days={stage_days}")
                if since is None:
                    base_days = cfg["harvester"]["bootstrap_days_default"].get(tf, 30)
                    days = bootstrap_overrides.get(tf, base_days)
                    since = ex.milliseconds() - 86_400_000 * int(days)
                limit = 1000
                all_rows: List[Dict[str, Any]] = []
                last_ts = since
                try:
                    while True:
                        buckets[ex_id].consume(1)
                        ohlcv = ex.fetch_ohlcv(market, tf_resolved, since=last_ts, limit=limit)
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
                try:
                    qa = validate_frame(df, tf_ms, max_gap_pct=max_gap_pct_accept)
                except Exception as e:
                    logger.opt(exception=True).error(f"{ex_id} {tf} {canonical} validation error: {e}")
                    state.r.hincrby("harvest:errors", ex_id, 1)
                    continue

                # Gap repair attempt if accepted but gaps remain OR if degraded due to gaps within accept threshold
                if qa.get('gaps',0) > 0 and not qa['degraded']:
                    repair_rows = _repair_gaps(ex, market, tf_resolved, tf_ms, qa['df'], buckets[ex_id])
                    if repair_rows:
                        merged = pd.concat([qa['df'], pd.DataFrame(repair_rows)]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                        try:
                            qa2 = validate_frame(merged, tf_ms, max_gap_pct=max_gap_pct_accept)
                            qa = qa2
                        except Exception:
                            pass
                elif qa['degraded'] and qa.get('gap_ratio',1.0) <= max_gap_pct_accept:
                    repair_rows = _repair_gaps(ex, market, tf_resolved, tf_ms, qa['df'], buckets[ex_id])
                    if repair_rows:
                        merged = pd.concat([qa['df'], pd.DataFrame(repair_rows)]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                        try:
                            qa2 = validate_frame(merged, tf_ms, max_gap_pct=max_gap_pct_accept)
                            qa = qa2
                        except Exception:
                            pass

                actual = len(qa["df"]) if qa.get("df") is not None else 0
                coverage = (actual / expected_rows) if (expected_rows and expected_rows>0) else 0.0
                state.hset("harvest:coverage", f"{ex_id}:{tf}:{canonical}", json.dumps({
                    "expected": expected_rows,
                    "actual": actual,
                    "coverage": round(coverage,4),
                    "gaps": qa["gaps"],
                    "gap_ratio": round(float(qa.get("gap_ratio",0.0)),4),
                    "accepted": not qa["degraded"],
                }))
                if qa["degraded"]:
                    logger.warning(f"Reject {ex_id} {tf} {canonical}: gaps={qa['gaps']} gap_ratio={qa.get('gap_ratio'):.3f} coverage={coverage:.2f}")
                    continue
                out = _out_path(base_dir, ex_id, tf, canonical)
                atomic_write_parquet(qa["df"], out, {"rows_added": len(qa["df"]), "gaps_detected": qa["gaps"],
                                                     "dupes_removed": qa["dupes"], "last_ts": int(qa["df"]['timestamp'].max().timestamp()*1000),
                                                     "degraded": False, "expected_rows": expected_rows, "coverage": coverage,
                                                     "gap_ratio": qa.get("gap_ratio", 0.0), "staged_days": stage_days})
                ckp.set(ex_id, canonical, tf, int(qa["df"]['timestamp'].max().timestamp()*1000))
                state.r.hincrby(f"harvest:bars_written:{ex_id}:{tf}", "count", len(qa["df"]))
                state.hset("harvest:checkpoint", f"{ex_id}:{tf}:{canonical}", int(qa["df"]['timestamp'].max().timestamp()*1000))
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

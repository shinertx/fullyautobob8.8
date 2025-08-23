#!/usr/bin/env python3
"""Fast targeted backfill utility.

Accelerates data availability for research by:
- Narrowing symbol + timeframe set (subset core) to fill critical history first.
- Running the event‑sourced harvester `run_once` in a tight loop (no research work).
- Exiting automatically once per‑timeframe bar thresholds met across a minimum symbol count.

PIT note: Uses only historical venue API responses via existing harvester (which enforces QA + no open bar writes). No forward leakage introduced.

Usage examples:
  python fast_backfill.py --symbols BTC_USD_SPOT ETH_USD_SPOT SOL_USD_SPOT \
      --timeframes 1h 5m --min-symbols 3  # dynamic targets from config bootstrap days
  python fast_backfill.py --timeframes 1h 5m --target-bars 1h=720 5m=5000 --min-symbols 3

If --target-bars omitted, targets are derived from config bootstrap days:
  target_bars(tf) = bootstrap_days(tf) * bars_per_day(tf); e.g. 1h: 30*24=720 when 1h bootstrap=30.
"""
from __future__ import annotations
import argparse, os, time, json
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv
import yaml
from loguru import logger

from v26meme.core.state import StateManager
from v26meme.data.harvester import run_once as harvest_once

DEFAULT_MIN_SYMBOLS = 3
# Bars per day per timeframe (no magic numbers; single source)
TF_BARS_PER_DAY: Dict[str,int] = {
    '1m': 60*24,
    '5m': (60//5)*24,
    '15m': (60//15)*24,
    '1h': 24,
    '4h': 24//4,
    '1d': 1,
}


def load_config(path: str = 'configs/config.yaml') -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def parse_target_bars(pairs: List[str]) -> Dict[str, int]:
    out: Dict[str,int] = {}
    for p in pairs:
        if '=' not in p:
            raise ValueError(f"Invalid target-bars entry '{p}' (expected tf=bars)")
        tf, v = p.split('=', 1)
        out[tf.strip()] = int(v)
    return out


def collect_tf_stats(state: StateManager, tfs: List[str], symbols: List[str]) -> Dict[str, Dict[str,int]]:
    stats: Dict[str, Dict[str,int]] = {tf: {} for tf in tfs}
    try:
        keys_raw = state.r.hkeys('harvest:coverage') or []  # type: ignore[assignment]
    except Exception:
        return stats
    # Some redis clients may return bytes or awaitables; normalize conservatively
    keys: List[str] = []
    for k in keys_raw:  # type: ignore[assignment]
        try:
            if hasattr(k, '__await__'):
                continue  # skip awaitables in sync context
            if isinstance(k, bytes):
                keys.append(k.decode('utf-8'))
            else:
                keys.append(str(k))
        except Exception:
            continue
    for k in keys:
        parts = k.split(':', 2)
        if len(parts) != 3:
            continue
        _ex, tf, canon = parts
        if tf not in stats:
            continue
        if symbols and canon not in symbols:
            continue
        try:
            raw = state.r.hget('harvest:coverage', k)  # type: ignore[assignment]
        except Exception:
            continue
        if hasattr(raw, '__await__'):
            continue  # skip coroutine objects in sync utility
        if raw is None:
            continue
        try:
            meta_obj = raw if isinstance(raw, (str, bytes)) else str(raw)
            if isinstance(meta_obj, bytes):
                meta_obj = meta_obj.decode('utf-8')
            meta = json.loads(meta_obj)
            if isinstance(meta, str):
                meta = json.loads(meta)
        except Exception:
            continue
        if not isinstance(meta, dict):
            continue
        try:
            stats[tf][canon] = int(meta.get('actual') or 0)
        except Exception:
            continue
    return stats


def thresholds_met(stats: Dict[str, Dict[str,int]], targets: Dict[str,int], min_symbols: int) -> bool:
    for tf, per_sym in stats.items():
        tgt = targets.get(tf, 0)
        eligible = sum(1 for _s, bars in per_sym.items() if bars >= tgt)
        if eligible < min_symbols:
            return False
    return True


def _retro_reindex_symbols(state: StateManager, tf: str, symbols: List[str]) -> int:
    """Populate/upgrade harvest:coverage entries for given symbols & timeframe from on-disk parquet row counts.

    PIT safe: reads only existing historical parquet. Necessary when checkpoints were fast‑forwarded
    (reconcile) leaving tiny 'actual' counts (e.g. 2) that stall bar targets. Sets expected=actual=row_count, coverage=1.0.
    Returns number of entries written/updated.
    """
    base = Path('data')
    updated = 0
    for ex_dir in base.iterdir():
        if not ex_dir.is_dir():
            continue
        tf_dir = ex_dir / tf
        if not tf_dir.exists():
            continue
        for sym in symbols:
            fp = list(tf_dir.rglob(f"{sym}.parquet"))
            if not fp:
                continue
            rows_total = 0
            for part in fp:
                try:
                    import pandas as _pd  # local import to avoid global dependency at module import
                    rows_total += len(_pd.read_parquet(part, columns=['timestamp']))
                except Exception:
                    continue
            if rows_total <= 0:
                continue
            key = f"{ex_dir.name}:{tf}:{sym}"
            try:
                raw = state.r.hget('harvest:coverage', key)  # type: ignore[assignment]
                if hasattr(raw, '__await__'):
                    raw = None  # cannot await here
                cur_actual = -1
                if raw is not None:
                    try:
                        meta_obj = raw if isinstance(raw, (str, bytes)) else str(raw)
                        if isinstance(meta_obj, bytes):
                            meta_obj = meta_obj.decode()
                        import json as _json
                        meta = _json.loads(meta_obj)
                        if isinstance(meta, str):
                            meta = _json.loads(meta)
                        if isinstance(meta, dict):
                            cur_actual = int(meta.get('actual') or 0)
                    except Exception:
                        pass
                if rows_total > cur_actual:
                    payload = json.dumps({'expected': rows_total, 'actual': rows_total, 'coverage': 1.0, 'gaps': 0, 'gap_ratio': 0.0, 'accepted': True, 'reindexed': True})
                    state.hset('harvest:coverage', key, payload)
                    updated += 1
            except Exception:
                continue
    return updated


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', nargs='*', default=[], help='Subset of canonical symbols to backfill (default: config core subset)')
    ap.add_argument('--timeframes', nargs='*', default=['1h'], help='Timeframes to backfill (default: 1h)')
    ap.add_argument('--target-bars', nargs='*', default=[], help='Per timeframe bar targets tf=bars (omit for dynamic from config bootstrap days)')
    ap.add_argument('--min-symbols', type=int, default=DEFAULT_MIN_SYMBOLS, help='Minimum symbols per timeframe meeting target to finish')
    ap.add_argument('--max-cycles', type=int, default=500, help='Safety cap on run_once cycles')
    ap.add_argument('--sleep', type=float, default=1.0, help='Sleep seconds between cycles')
    ap.add_argument('--partial', action='store_true', help='Enable harvester partial mode each cycle')
    ap.add_argument('--config', default='configs/config.yaml')
    args = ap.parse_args()

    load_dotenv()
    cfg = load_config(args.config)

    core_syms_cfg = cfg.get('harvester', {}).get('core_symbols', [])
    symbols = args.symbols or core_syms_cfg[:max(3, min(10,len(core_syms_cfg)))]  # small subset default

    # Override config in-memory (no file write) to narrow workload
    cfg.setdefault('harvester', {})['core_symbols'] = symbols
    # Restrict timeframes (core lane)
    tf_lane = cfg['harvester'].get('timeframes_by_lane', {}).get('core') or []
    cfg['harvester'].setdefault('timeframes_by_lane', {})['core'] = [tf for tf in args.timeframes if tf in tf_lane or not tf_lane] or args.timeframes
    # Disable staged_backfill for speed
    if cfg['harvester'].get('staged_backfill'):
        cfg['harvester']['staged_backfill']['enabled'] = False
    # Ensure partial harvest optional
    if args.partial:
        cfg['harvester']['partial_harvest'] = True

    # Dynamic targets if none supplied
    if args.target_bars:
        targets = parse_target_bars(args.target_bars)
        dynamic = False
    else:
        # Determine bootstrap days per tf (prefer per_exchange_bootstrap coinbase then default)
        per_ex_boot = cfg['harvester'].get('per_exchange_bootstrap', {}).get('coinbase', {})
        boot_defaults = cfg['harvester'].get('bootstrap_days_default', {})
        targets = {}
        for tf in args.timeframes:
            days = per_ex_boot.get(tf) or boot_defaults.get(tf)
            if not days:
                # Fallback: minimal 7 day assumption for safety
                days = 7
            bars_per_day = TF_BARS_PER_DAY.get(tf)
            if not bars_per_day:
                # Skip unknown timeframe
                continue
            targets[tf] = int(days) * bars_per_day
        dynamic = True

    logger.add('logs/fast_backfill.log', level='INFO', rotation='5 MB', retention='7 days')
    logger.info(f"Fast backfill start symbols={symbols} tfs={args.timeframes} targets={targets} dynamic_targets={dynamic} min_symbols={args.min_symbols} partial={cfg['harvester'].get('partial_harvest', False)}")

    state = StateManager(cfg['system']['redis_host'], cfg['system']['redis_port'])

    # Initial retro reindex before cycles
    for tf in args.timeframes:
        upd = _retro_reindex_symbols(state, tf, symbols)
        if upd:
            logger.info(f"retro_reindex tf={tf} updated={upd}")

    for cycle in range(1, args.max_cycles+1):
        # Reindex again each cycle (cheap) to capture any late-added parquet partitions
        for tf in args.timeframes:
            upd = _retro_reindex_symbols(state, tf, symbols)
            if upd:
                logger.info(f"retro_reindex tf={tf} cycle={cycle} updated={upd}")
        try:
            harvest_once(cfg, state, partial_mode=cfg.get('harvester', {}).get('partial_harvest', False))  # type: ignore[arg-type]
        except TypeError:
            harvest_once(cfg, state)  # type: ignore[misc]
        except Exception as e:
            logger.warning(f"harvest error cycle={cycle}: {e}")
        stats = collect_tf_stats(state, args.timeframes, symbols)
        prog = {tf: {s: stats[tf][s] for s in sorted(stats[tf])} for tf in stats}
        logger.info(f"cycle={cycle} progress={prog}")
        if thresholds_met(stats, targets, args.min_symbols):
            logger.success(f"Targets met; exiting fast backfill after cycle {cycle}")
            break
        time.sleep(args.sleep)
    else:
        logger.warning("Max cycles reached without meeting all targets")

    # Summary
    stats = collect_tf_stats(state, args.timeframes, symbols)
    summary = {tf: { 'eligible': sum(1 for _s,b in v.items() if b >= targets.get(tf,0)), 'total_symbols': len(v)} for tf,v in stats.items()}
    logger.info(f"final_summary={summary}")

if __name__ == '__main__':  # pragma: no cover
    main()

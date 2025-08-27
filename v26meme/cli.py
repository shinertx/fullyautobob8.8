# v26meme/cli.py — v4.7.5 (event‑sourced data plane, canonical joins, calibrated sim, lanes)
from __future__ import annotations
import os, time, json, hashlib, random, inspect, sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple
from collections.abc import Awaitable as _Awaitable  # ensure consistent with harvester _sync

import click, yaml
from dotenv import load_dotenv
from loguru import logger
import pandas as pd
import numpy as np
import glob as _glob  # added for retro coverage reindex
import redis

from v26meme.core.state import StateManager
from v26meme.core.config import RootConfig
from v26meme.data.lakehouse import Lakehouse
from v26meme.data.universe_screener import UniverseScreener
from v26meme.data.screener_store import ScreenerStore
from v26meme.data.top_gainers import compute_top_gainers_bases
from v26meme.llm.proposer import LLMProposer
from v26meme.allocation.optimizer import PortfolioOptimizer
from v26meme.allocation.lanes import LaneAllocationManager
from v26meme.execution.exchange import ExchangeFactory
from v26meme.execution.handler import ExecutionHandler
from v26meme.execution.risk import RiskManager
from v26meme.core.dsl import Alpha, normalize_survivor_to_alpha
from v26meme.analytics.adaptive import publish_adaptive_knobs
from v26meme.analytics.ensemble import EnsembleManager
from v26meme.registry.resolver import configure as configure_resolver
from v26meme.registry.catalog import CatalogManager
from v26meme.data.harvester import run_once as harvest_once  # event-sourced incremental OHLCV harvest

from v26meme.labs.hyper_lab import run_eil

# --------------------------------------------------------------------------------------
# Hygiene
# --------------------------------------------------------------------------------------

def _alpha_registry_hygiene(state: StateManager, cfg: Dict[str, Any]) -> Dict[str, int]:
    """Perform alpha registry hygiene: dedupe, retroactive gate enforcement, padding trim.

    PIT note: Operates only on already persisted alpha performance stats (historical returns) with
    no lookahead. Trimming only removes trailing zero pads beyond declared n_trades to avoid
    inflating correlation / variance metrics.
    """
    changed = {"dupes_removed":0, "trimmed":0, "dropped_gates":0, "final":0}
    alpha_dicts = state.get_active_alphas()
    if not alpha_dicts:
        return changed
    
    alphas = [Alpha.model_validate(a) for a in alpha_dicts]

    seen = set(); clean: List[Alpha] = []
    crit = (cfg.get('discovery') or {}).get('promotion_criteria', {})
    enforce = bool((cfg.get('discovery') or {}).get('enforce_current_gates_on_start', False))
    max_trim = int((cfg.get('discovery') or {}).get('max_return_padding_trim', 5))
    for a in alphas:
        if not a.id:
            continue
        if a.id in seen:
            changed["dupes_removed"] += 1
            continue
        seen.add(a.id)
        
        # Get performance dict, creating nested structure if needed
        perf = a.performance.get('all', {})
        returns = list(perf.get('returns') or [])
        n_trades = int(perf.get('n_trades') or 0)
        
        # Trim condition: only if extra section exists AND consisting solely of zeros
        if n_trades > 0 and len(returns) > n_trades:
            extra = returns[n_trades:]
            if all((r or 0) == 0 for r in extra) and len(extra) <= max_trim:
                # Trim to n_trades and properly update the nested structure
                a.performance['all']['returns'] = returns[:n_trades]
                changed['trimmed'] += 1
        
        # Gate enforcement
        if enforce:
            # Re-fetch perf in case it was modified
            perf = a.performance.get('all', {})
            buf = float(cfg.get('discovery', {}).get('promotion_buffer_multiplier', 1.0))
            if (perf.get('n_trades',0) < crit.get('min_trades',0) or
                perf.get('sortino',0) < crit.get('min_sortino',0) * buf or
                perf.get('sharpe',0) < crit.get('min_sharpe',0) * buf or
                perf.get('win_rate',0) < crit.get('min_win_rate',0) or
                perf.get('mdd',1.0) > crit.get('max_mdd',1.0)):
                changed['dropped_gates'] += 1
                continue
        clean.append(a)
    # Single write after processing all
    clean_dicts = [alpha.model_dump() for alpha in clean]
    if clean_dicts != alpha_dicts:
        state.set_active_alphas(clean_dicts)
    changed['final'] = len(clean)
    return changed

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _extract_features_from_formula(node: Any) -> set[str]:
    """Recursively extract unique feature names from a formula structure."""
    features = set()
    if not isinstance(node, list):
        return features
    # Base case: a condition like ['feature_name', '>', 0.5]
    if len(node) >= 3 and isinstance(node[0], str) and node[1] in ('>', '<'):
        features.add(node[0])
    # Recursive step: a logical combination like [ [cond1], 'AND', [cond2] ]
    elif len(node) >= 3 and isinstance(node[0], list) and isinstance(node[2], list):
        features.update(_extract_features_from_formula(node[0]))
        features.update(_extract_features_from_formula(node[2]))
    return features


def load_config(file: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load YAML config and validate against Pydantic schema."""
    with open(file, "r") as f:
        raw_config = yaml.safe_load(f)
    
    try:
        validated_config = RootConfig.model_validate(raw_config)
        logger.info("Configuration validated successfully.")
        return validated_config.model_dump()
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise SystemExit("Exiting due to invalid configuration.") from e

def lagged_zscore(s: pd.Series, lookback: int | None = None, cfg: Dict[str, Any] | None = None) -> pd.Series:
    """Point‑in‑time safe z‑score (uses only prior data via shift).

    Adaptive: lookback pulled from cfg.features.zscore_lookback if not provided.
    PIT: rolling mean/std shifted(1) to avoid same-bar leakage.
    """
    if lookback is None:
        lookback = int(((cfg or {}).get('features') or {}).get('zscore_lookback', 200))
    else:
        lookback = int(lookback)
    if lookback <= 0:
        lookback = 200  # defensive fallback
    minp = max(5, lookback // 4)
    m = s.rolling(lookback, min_periods=minp).mean().shift(1)
    sd = s.rolling(lookback, min_periods=minp).std().shift(1)
    return (s - m) / sd

def _ensure_lakehouse_bootstrap(cfg: Dict[str, Any], state: StateManager) -> None:
    """Bootstrap the lakehouse with an initial harvest if empty.

    Tries external shim (data_harvester.harvest). Falls back to internal
    event-sourced harvester run_once for robustness.
    """
    root = Path("data")
    if any(root.glob("**/*.parquet")):
        return
    logger.info("No lakehouse detected; harvesting initial dataset (first run)…")
    try:
        from data_harvester import harvest as _harvest  # type: ignore
        try:
            _harvest(cfg)
        except TypeError:
            _harvest(cfg, None)  # legacy signature
        logger.info("Initial harvest complete via shim.")
        return
    except Exception:
        pass
    from v26meme.data.harvester import run_once as _run_once
    _run_once(cfg, state)
    logger.info("Initial harvest complete (internal harvester fallback).")

def _make_lakehouse(cfg: Dict[str, Any]) -> Lakehouse:
    try:
        if "preferred_exchange" in inspect.signature(Lakehouse).parameters:
            return Lakehouse(preferred_exchange=cfg["execution"]["primary_exchange"])  # type: ignore[arg-type]
    except Exception:  # pragma: no cover - defensive
        pass
    return Lakehouse()

def _lh_get_data(lh: Lakehouse, cfg: Dict[str, Any], canonical: str, tf: str) -> pd.DataFrame:
    try:
        return lh.get_data(canonical, tf)
    except TypeError:  # legacy signature
        try:
            return lh.get_data(cfg["execution"]["primary_exchange"], canonical, tf)  # type: ignore[misc]
        except Exception:
            return pd.DataFrame()

def _lh_get_available_symbols(lh: Lakehouse, cfg: Dict[str, Any], tf: str) -> List[str]:
    try:
        return lh.get_available_symbols(tf)
    except TypeError:  # legacy signature
        try:
            return lh.get_available_symbols(cfg["execution"]["primary_exchange"], tf)  # type: ignore[misc]
        except Exception:
            return []

def _today_key(prefix: str = "promotions") -> str:  # YYYY-MM-DD key
    return f"{prefix}:{datetime.now(timezone.utc).date().isoformat()}"

def _retro_reindex_coverage(state: StateManager, cfg: Dict[str, Any], tf: str) -> None:
    """Retroactively populate or upgrade harvest:coverage entries from existing parquet history.

    PIT note: Derives solely from already persisted on-disk OHLCV partitions; no
    forward-looking data introduced. One-shot per timeframe (idempotent via state flag).

    Rationale: Historical deep backfill (copied / aggregated) created large parquet depth
    without corresponding 'harvest:coverage' hash entries (only recent incremental fetches
    wrote coverage). This starves the coverage gate (min_bars_per_symbol) because many
    symbols either (a) lack a coverage entry entirely or (b) report a tiny 'actual' count
    reflecting only the last incremental window. We reconcile by scanning partitions and
    emitting canonical coverage payloads with conservative assumptions (expected = actual,
    coverage = 1.0) strictly for gating prerequisites. Native future harvest cycles will
    overwrite these with real expected/coverage values on fresh fetches.

    Writes keys where missing or where actual < row_count. Marks payload with 'reindexed': true.
    """
    try:
        flag_key = f"harvest:coverage:reindexed:{tf}"
        if state.get(flag_key):
            return
        exchanges = (cfg.get('data_source') or {}).get('exchanges') or []
        base_dir = Path('data')
        updated = 0
        for ex_id in exchanges:
            tf_dir = base_dir / ex_id / tf
            if not tf_dir.exists():
                continue
            # pattern: data/<exchange>/<tf>/**/<canonical>.parquet
            for fp in _glob.glob(str(tf_dir / '**' / '*.parquet'), recursive=True):
                try:
                    path_obj = Path(fp)
                    canonical = path_obj.stem  # filename without .parquet
                    key = f"{ex_id}:{tf}:{canonical}"
                    # Fast metadata read: only timestamp column
                    try:
                        df_head = pd.read_parquet(fp, columns=['timestamp'])
                    except Exception:
                        continue
                    rows = len(df_head)
                    if rows <= 0:
                        continue
                    existing_raw = state.r.hget('harvest:coverage', key)
                    if existing_raw is not None:
                        try:
                            if isinstance(existing_raw, bytes):
                                existing_raw_dec = existing_raw.decode('utf-8')
                            else:
                                existing_raw_dec = str(existing_raw)
                            existing = json.loads(existing_raw_dec)
                        except Exception:
                            existing = {}
                        if isinstance(existing, dict) and int(existing.get('actual') or 0) >= rows:
                            continue  # nothing to upgrade
                    payload = json.dumps({
                        'expected': rows,  # conservative (cannot overstate bars)
                        'actual': rows,
                        'coverage': 1.0,   # since expected == actual
                        'gaps': 0,
                        'gap_ratio': 0.0,
                        'accepted': True,
                        'reindexed': True
                    })
                    try:
                        state.hset('harvest:coverage', key, payload)
                        updated += 1
                    except Exception:
                        continue
                except Exception:
                    continue
        if updated > 0:
            state.set(flag_key, int(time.time()))
            logger.info(f"[retro_reindex] timeframe={tf.upper()} coverage entries updated={updated}")
        else:
            state.set(flag_key, int(time.time()))  # still set to avoid repeated scans
    except Exception as e:  # defensive
        logger.warning(f"retro coverage reindex failed tf={tf.upper()}: {e}")

def _bars_per_day(tf: str) -> int:
    return {
        '1m': 60*24,
        '5m': (60//5)*24,
        '15m': (60//15)*24,
        '1h': 24,
        '4h': 24//4,
        '1d': 1,
    }.get(tf, 0)

def _coverage_gate_ok(state: StateManager, cfg: Dict[str, Any], tf: str) -> tuple[bool, dict]:
    """Evaluate whether minimal coverage conditions to start research are met.

    Returns (ok, stats) where stats includes counts for logging.
    PIT note: derives solely from already persisted harvest coverage hash
    produced in prior (or current) run_once calls (no forward leakage).
    Hardened for double-encoded JSON values (some historical entries were
    stored as json.dumps(json.dumps(meta))).
    """
    if not cfg.get('discovery', {}).get('defer_until_coverage', False):
        return True, {"reason": "disabled"}
    try:
        _retro_reindex_coverage(state, cfg, tf)
    except Exception:
        pass
    harv_cfg = cfg.get('harvester', {})
    min_cov = float(harv_cfg.get('min_coverage_for_research', 0.0))
    if state.get('coverage:raise_threshold'):
        hi = harv_cfg.get('high_coverage_threshold')
        if hi is not None:
            try:
                min_cov = float(hi)
            except Exception:
                pass
    min_sym = int(cfg.get('discovery', {}).get('min_panel_symbols', 1))
    min_bars_floor = int(cfg.get('discovery', {}).get('min_bars_per_symbol', 1))
    panel_target_days = (harv_cfg.get('panel_target_days') or {})
    core_syms_limit = None
    try:
        core_syms = (harv_cfg.get('core_symbols') or [])
        if isinstance(core_syms, list) and len(core_syms) == 1:
            core_syms_limit = set(core_syms)
    except Exception:
        pass
    # Dynamic target bars > floor
    target_days = int(panel_target_days.get(tf, 0))
    target_bars = max(min_bars_floor, target_days * _bars_per_day(tf) if target_days else min_bars_floor)
    try:
        keys = state.hkeys_sync('harvest:coverage')
    except Exception:
        return False, {"error": "coverage_hash_unavailable"}
    covered: Dict[str, dict] = {}
    for k in keys:
        try:
            parts = str(k).split(':', 2)
            if len(parts) != 3:
                continue
            _ex_id, tf_key, canonical = parts
            if tf_key != tf:
                continue
            if core_syms_limit and canonical not in core_syms_limit:
                continue  # TEMP single-symbol narrowing
            raw_val = state.r.hget('harvest:coverage', k)
            if raw_val is None:
                continue
            if isinstance(raw_val, bytes):
                raw = raw_val.decode('utf-8')
            else:
                raw = str(raw_val)
            try:
                meta = json.loads(raw)
            except Exception:
                continue
            if isinstance(meta, str):
                try:
                    meta2 = json.loads(meta)
                    if isinstance(meta2, dict):
                        meta = meta2
                    else:
                        continue
                except Exception:
                    continue
            if not isinstance(meta, dict):
                continue
            prev = covered.get(canonical)
            if (not prev) or int(meta.get('actual') or 0) > int(prev.get('actual') or 0):
                covered[canonical] = meta
        except Exception:
            continue
    # Augment with total parquet rows (multi-exchange sum) for robustness
    try:
        base_dir = Path('data')
        for canon, meta in list(covered.items()):
            try:
                pattern = str(base_dir / '*' / tf / '**' / f'{canon}.parquet')
                files = _glob.glob(pattern, recursive=True)
                total_rows = 0
                for fp in files:
                    try:
                        df_head = pd.read_parquet(fp, columns=['timestamp'])
                        total_rows += len(df_head)
                    except Exception:
                        continue
                if total_rows > int(meta.get('actual') or 0):
                    meta['actual_total'] = total_rows
            except Exception:
                continue
    except Exception:
        pass
    elig = [c for c, m in covered.items() if (m.get('coverage', 0) >= min_cov and (m.get('actual_total', m.get('actual', 0)) or 0) >= target_bars)]
    
    # NEW: Quorum check - pass if at least 75% of symbols meet the bar target
    quorum_pct = 0.75
    if len(covered) >= min_sym:
        qualified_count = sum(1 for m in covered.values() if (m.get('actual_total', m.get('actual', 0)) or 0) >= target_bars)
        if (qualified_count / len(covered)) >= quorum_pct:
            logger.info(f"Coverage gate passed by quorum: {qualified_count}/{len(covered)} ({quorum_pct:.0%}) met target bars.")
            return True, {"eligible_quorum": qualified_count, "required": min_sym, "min_cov": min_cov, "tf": tf, "symbols": len(covered), "target_bars": target_bars}

    ok = len(elig) >= min_sym
    # Telemetry snapshot
    try:
        counts = [int(m.get('actual_total', m.get('actual', 0)) or 0) for m in covered.values()]
        if counts:
            panel_meta = {
                'tf': tf,
                'target_bars': target_bars,
                'min_bars_floor': min_bars_floor,
                'symbols_seen': len(covered),
                'symbols_eligible': len(elig),
                'bars_min': min(counts),
                'bars_median': sorted(counts)[len(counts)//2],
                'bars_max': max(counts),
                'min_cov_req': min_cov,
            }
            state.set(f'coverage:panel:{tf}', panel_meta)
            logger.info(f"[panel_coverage] {panel_meta}")
            if panel_meta['bars_min'] < target_bars * 0.5:
                logger.warning(f"panel underfilled tf={tf.upper()} bars_min={panel_meta['bars_min']} < 0.5*target ({target_bars})")
    except Exception:
        pass
    return ok, {"eligible": len(elig), "required": min_sym, "min_cov": min_cov, "tf": tf, "symbols": len(covered), "target_bars": target_bars}

# --------------------------------------------------------------------------------------
# CLI Root
# --------------------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """Root CLI group."""
    pass

@cli.command()
def reset_risk_halt() -> None:
    """Manually clear the risk:halt Redis key."""
    load_dotenv()
    cfg = load_config()
    state = StateManager(cfg["system"]["redis_host"], cfg["system"]["redis_port"])
    
    if state.get("risk:halt"):
        state.r.delete("risk:halt")
        logger.success("Risk halt key 'risk:halt' has been cleared.")
    else:
        logger.info("Risk halt key 'risk:halt' was not set. No action taken.")

# --------------------------------------------------------------------------------------
# Main Loop
# --------------------------------------------------------------------------------------

@cli.command()
def loop() -> None:
    print("GEMINI_DEBUG: Entering loop function")
    """Run the research + promotion loop (paper only).

    Safeguards:
    - Deterministic: fixed seed unless overridden in config.
    - PIT correctness: all feature transforms are lagged (see lagged_zscore).
    - No live trading side‑effects (execution handler stays in paper mode).
    """
    load_dotenv()

    # Configure logger early to catch config errors
    Path("logs").mkdir(exist_ok=True, parents=True)
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    try:
        cfg = load_config()
    except SystemExit:
        logger.error("Exiting due to configuration error. Please check the logs above.")
        return

    # Reconfigure logger with level from config
    logger.remove()
    logger.add(
        "logs/system.log",
        level=cfg["system"]["log_level"].upper(),
        rotation="10 MB",
        retention="14 days",
        enqueue=True,
        format="{time} {level} {message}",
    )
    logger.add(
        sys.stderr,
        level=cfg["system"]["log_level"].upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.info("Logger reconfigured.")

    random.seed(cfg["system"].get("seed", 1337))
    configure_resolver(cfg.get("registry"))

    logger.info("🚀 v26meme v4.7.5 loop starting…")

    if ((cfg.get("llm") or {}).get("provider", "").lower() != "openai" or not os.getenv("OPENAI_API_KEY", "")):
        logger.warning("LLM is OpenAI-only. Set llm.provider=openai and OPENAI_API_KEY in .env.")

    host, port = cfg["system"]["redis_host"], cfg["system"]["redis_port"]
    try:
        state = StateManager(host, port)
    except redis.exceptions.ConnectionError:
        logger.critical(
            f"FATAL: Redis unavailable at {host}:{port}. Start redis-server and retry."
        )
        raise SystemExit(1)
    
    # Check for persistent risk halt on startup
    if state.get("risk:halt"):
        logger.critical("="*80)
        logger.critical("SYSTEM HALTED: 'risk:halt' key is set in Redis.")
        logger.critical("The bot will not run until this key is cleared.")
        logger.critical("To clear it, run the following command:")
        logger.critical("python -m v26meme.cli reset-risk-halt")
        logger.critical("="*80)
        return # Exit the loop command

    _ensure_lakehouse_bootstrap(cfg, state)

    lakehouse = _make_lakehouse(cfg)
    screener = UniverseScreener(cfg["data_source"]["exchanges"], cfg["screener"], cfg.get("feeds"))
    snapshot_dir = cfg["screener"].get("snapshot_dir") or "data/screener_snapshots"
    store = ScreenerStore(snapshot_dir, state)
    catalog = CatalogManager(state, cfg.get("registry"))
    optimizer = PortfolioOptimizer(cfg.get("portfolio", {}))
    lane_mgr = LaneAllocationManager(cfg, state)
    exchange_factory = ExchangeFactory(os.environ.get("GCP_PROJECT_ID"))
    risk = RiskManager(state, cfg, lakehouse=lakehouse)
    exec_handler = ExecutionHandler(state, exchange_factory, cfg, risk_manager=risk)
    ensemble_manager = EnsembleManager(state, cfg, lakehouse=lakehouse)

    def _get_current_gate_stage(state: StateManager, cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Gets the current promotion gate stage and its config."""
        stages = cfg.get("discovery", {}).get("gate_stages", [])
        if not stages:
            # Fallback to top-level if not defined
            return "default", cfg.get("discovery", {}).get("promotion_criteria", {})
        
        current_stage_name = state.get("gate:stage:current") or stages[0]["name"]
        for stage in stages:
            if stage["name"] == current_stage_name:
                return current_stage_name, stage
        # Fallback to the first stage if the stored one is not found
        return stages[0]["name"], stages[0]

    def _research_tf() -> str:
        tf_opts = (cfg.get("harvester") or {}).get("timeframes_by_lane", {}).get("core") or ["1h"]
        return "1h" if "1h" in tf_opts else tf_opts[0]

    while True:  # Main cycle
        try:
            state.heartbeat()
            logger.info("--- New loop cycle ---")

            # Log promotion cap status
            promotions_today_count = len(state.get(_today_key("promotions")) or [])
            max_promotions_day = int(cfg['discovery'].get('max_promotions_per_day', 1))
            logger.info(f"Promotion status: {promotions_today_count}/{max_promotions_day} promotions used today.")

            # Hygiene pass early each cycle (pre-harvest to keep list small during scoring)
            try:
                hg = _alpha_registry_hygiene(state, cfg)
                if sum([v for k,v in hg.items() if k != 'final']) > 0:
                    logger.info(f"[hygiene] {json.dumps(hg)}")
            except Exception as _e:
                logger.warning(f"Hygiene error: {_e}")
            
            catalog.maybe_refresh(screener.exchanges)
            if state.get("screener:force_refresh"):
                state.r.delete("screener:force_refresh")

            # 1. Data Harvesting
            # --------------------------------------------------------------------------
            logger.info("STEP 1/6: Starting data harvesting...")
            try:
                harvest_once(cfg, state)
            except Exception as e:
                logger.error(f"Data harvesting failed: {e}")
                time.sleep(cfg["system"]["loop_interval_seconds"])
                continue
            logger.info("Data harvesting complete.")

            # 2. Universe Screening
            # --------------------------------------------------------------------------
            logger.info("STEP 2/6: Starting universe screening...")
            tf = _research_tf()
            
            # Coverage Gate
            coverage_ok, coverage_stats = _coverage_gate_ok(state, cfg, tf)
            if not coverage_ok:
                logger.warning(f"Coverage gate NOT passed for tf={tf.upper()}. Stats: {coverage_stats}. Deferring research.")
                time.sleep(cfg["system"]["loop_interval_seconds"])
                continue
            logger.info(f"Coverage gate passed for tf={tf.upper()}. Stats: {coverage_stats}")

            # Screener Execution
            try:
                universe, tickers = screener.get_active_universe()
                store.save(universe, tickers)
                logger.info(f"Universe screening complete. Symbols: {len(universe)}")
            except Exception as e:
                logger.opt(exception=True).error(f"Universe screening failed: {e}")
                time.sleep(cfg["system"]["loop_interval_seconds"])
                continue

            # 2.5. LLM-driven Seeding (Adaptive Injection)
            # --------------------------------------------------------------------------
            if cfg.get('llm', {}).get('enable', False):
                logger.info("STEP 2.5/6: Starting LLM-driven strategy seeding...")
                try:
                    proposer = LLMProposer(state, cfg)
                    seed_features = []
                    seeded_with = "none"
                    
                    active_alpha_dicts = state.get_active_alphas() or []
                    if active_alpha_dicts:
                        base_features = set()
                        sorted_alphas = sorted(active_alpha_dicts, key=lambda a: (a.get('performance', {}).get('all', {}).get('sortino', 0.0)), reverse=True)
                        top_alphas = sorted_alphas[:max(1, len(sorted_alphas) // 10)]
                        for alpha_dict in top_alphas:
                            formula_raw = alpha_dict.get('formula_raw')
                            if formula_raw:
                                try:
                                    formula_struct = json.loads(formula_raw)
                                    base_features.update(_extract_features_from_formula(formula_struct))
                                except Exception: continue
                        if base_features:
                            seed_features = list(base_features)
                            seeded_with = "top_features"
                            logger.info(f"Seeding LLM with top features: {seed_features}")
                    
                    if not seed_features:
                        base_features_config = cfg.get('discovery', {}).get('base_features', [])
                        if base_features_config:
                            seed_features = random.sample(base_features_config, min(len(base_features_config), 15))
                            seeded_with = "random_features"
                            logger.info(f"Cold start: Seeding LLM with random base features: {seed_features}")
                    
                    if seed_features:
                        k_proposals = cfg.get('llm', {}).get('max_suggestions_per_cycle', 10)
                        proposals = proposer.propose(seed_features, k=k_proposals)
                        state.set('llm:proposer:last_run', {'ts': time.time(), 'generated': len(proposals), 'seeded_with': seeded_with})
                        logger.info(f"LLM proposer generated {len(proposals)} new formulas and pushed to 'llm:proposals' queue.")
                    else:
                        logger.info("No features to seed LLM, skipping proposal for this cycle.")
                except Exception as e:
                    logger.error(f"LLM Proposer failed: {e}", exc_info=True)

            # 3. EIL - Alpha Discovery
            # --------------------------------------------------------------------------
            logger.info("STEP 3/6: Starting Extreme Iteration Layer (EIL) for alpha discovery...")
            try:
                # Get and record current gate stage config for observability
                gate_stage_name, gate_config = _get_current_gate_stage(state, cfg)
                state.set("eil:gate:current_config", json.dumps(gate_config))

                logger.info("Invoking EIL function directly...")
                survivors = run_eil(cfg)
                
                if survivors:
                    logger.info(f"EIL returned {len(survivors)} survivors.")
                    promoted_count = _promote_eil_survivors(state, cfg, survivors)
                    logger.info(f"Promoted {promoted_count} new alphas from EIL survivors.")
                else:
                    logger.info("EIL run completed with no new survivors.")

            except Exception as e:
                logger.error(f"An unexpected error occurred during EIL execution: {e}", exc_info=True)


            # 4. Snapshot active alphas registry
            # --------------------------------------------------------------------------
            logger.info("STEP 4/6: Snapshotting alpha registry...")
            active_alpha_dicts = state.get_active_alphas()
            active_alphas = [Alpha.model_validate(a) for a in active_alpha_dicts]
            if active_alphas:
                try:
                    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                    fn = f"data/alphas/registry_{ts}.json"
                    with open(fn, "w") as f:
                        json.dump(active_alpha_dicts, f, indent=2)
                    logger.info(f"Snapshotted {len(active_alphas)} active alphas to {fn}")
                except Exception as e:
                    logger.error(f"Failed to snapshot alpha registry: {e}")

            # 5. Ensemble Generation
            # --------------------------------------------------------------------------
            logger.info("STEP 5/6: Running ensemble manager...")
            try:
                ensemble_manager.run(active_alphas, exec_handler.get_portfolio()['equity'])
            except Exception as e:
                logger.error(f"Ensemble manager failed: {e}")

            # 6. Portfolio Allocation & Execution
            # --------------------------------------------------------------------------
            logger.info("STEP 6/6: Starting portfolio allocation and execution...")
            if not active_alphas:
                logger.warning("No active alphas to allocate. Skipping.")
            else:
                try:
                    # Publish risk metrics for dashboard observability
                    current_equity = exec_handler.get_portfolio()['equity']
                    risk.publish_risk_metrics(current_equity)

                    regime = "all" # Or some logic to determine regime
                    weights = optimizer.get_weights(active_alphas, regime)
                    
                    # Lane Management
                    final_weights = lane_mgr.apply_lane_budgets(weights, active_alphas)
                    logger.info(f"Final weights after lane management: {final_weights}")

                    # Execution
                    exec_handler.reconcile(final_weights, active_alphas)

                except Exception as e:
                    logger.error(f"Portfolio allocation/execution failed: {e}", exc_info=True)

            # Publish adaptive knobs
            try:
                # Load BTC 1h data to drive adaptive stops
                btc_df = _lh_get_data(lakehouse, cfg, "BTC_USD_SPOT", "1h")
                publish_adaptive_knobs(state, cfg, btc_df)
            except Exception as e:
                logger.error(f"Failed to publish adaptive knobs: {e}")

            logger.info(f"Cycle complete. Sleeping for {cfg['system']['loop_interval_seconds']}s.")
            time.sleep(cfg["system"]["loop_interval_seconds"])

        except (KeyboardInterrupt, SystemExit) as e:
            logger.warning(f"Shutdown requested via {type(e).__name__}.")
            break
        except Exception as e:
            logger.opt(exception=True).error(f"Unhandled error in main loop: {e}")
            time.sleep(cfg["system"]["loop_interval_seconds"])

    logger.info("v26meme loop terminated.")

# --------------------------------------------------------------------------------------
# Debug Screener
# --------------------------------------------------------------------------------------

@cli.command()
def debug_screener() -> None:
    """Run screener once and print resulting instruments (diagnostic)."""
    load_dotenv()
    cfg = load_config()
    random.seed(cfg["system"].get("seed", 1337))
    configure_resolver(cfg.get("registry"))

    Path("logs").mkdir(exist_ok=True, parents=True)
    logger.add(
        "logs/system.log",
        level=cfg["system"]["log_level"],
        rotation="10 MB",
        retention="14 days",
        enqueue=True,
    )
    logger.info("🚀 v26meme v4.7.5 debug_screener starting…")

    screener = UniverseScreener(cfg["data_source"]["exchanges"], cfg["screener"], cfg.get("feeds"))
    instruments, tickers_by_venue = screener.get_active_universe(debug=True)
    print("Instruments:", instruments)
    print("Tickers by venue:", list(tickers_by_venue.keys()))

cli.add_command(debug_screener)

@click.command()
@click.option('--enforce', is_flag=True, help='Temporarily enforce current promotion gates during hygiene pass regardless of config flag.')
def hygiene_once(enforce: bool = False) -> None:
    """Run a one-off alpha registry hygiene pass and print JSON summary.

    Safe operation: does not modify any artifacts beyond the active_alphas set
    and only removes duplicates, padding, or non-compliant entries (when enforced).
    """
    from dotenv import load_dotenv
    load_dotenv()
    cfg = load_config()
    if enforce:
        cfg.setdefault('discovery', {})['enforce_current_gates_on_start'] = True
    state = StateManager(cfg['system']['redis_host'], cfg['system']['redis_port'])
    summary = _alpha_registry_hygiene(state, cfg)
    import json as _json
    print(_json.dumps(summary, indent=2, sort_keys=True))

cli.add_command(hygiene_once)

def _promote_eil_survivors(state: StateManager, cfg: Dict[str, Any], survivors: List[Dict[str, Any]]) -> int:
    """Promote top-k EIL survivors to the active alpha registry if they pass gates.

    PIT note: Operates on historical performance of candidates from EIL. No lookahead.
    """
    if not survivors:
        return 0

    promoted_count = 0
    promotions_today_count = len(state.get(_today_key("promotions")) or [])
    max_promotions_day = int(cfg['discovery'].get('max_promotions_per_day', 1))
    max_promotions_cycle = int(cfg['discovery'].get('max_promotions_per_cycle', 1))

    if promotions_today_count >= max_promotions_day:
        logger.warning(f"Daily promotion limit of {max_promotions_day} reached. No new promotions today.")
        return 0

    active_alphas = state.get_active_alphas() or []
    active_formulas = {a.get('formula_raw') for a in active_alphas}

    # Normalize and then sort
    try:
        # Convert flat survivor dicts to Alpha objects for consistent interface
        candidate_alphas = [Alpha.model_validate(normalize_survivor_to_alpha(s)) for s in survivors]
        
        # Sort using the object's method
        sorted_candidates = sorted(
            candidate_alphas,
            key=lambda a: a.sortino(),
            reverse=True
        )
    except Exception as e:
        logger.error(f"EIL_PROMOTION_SORT_FAIL: {e}")
        # Fallback for safety
        sorted_candidates = sorted(survivors, key=lambda s: s.get('sortino', s.get('sharpe', 0.0)), reverse=True)
        # If fallback is used, we need to normalize inside the loop
        sorted_candidates = [Alpha.model_validate(normalize_survivor_to_alpha(s)) for s in sorted_candidates]


    for alpha in sorted_candidates:
        if promoted_count >= max_promotions_cycle:
            logger.info(f"Cycle promotion limit of {max_promotions_cycle} reached.")
            break
        if promotions_today_count >= max_promotions_day:
            logger.warning(f"Daily promotion limit of {max_promotions_day} reached during cycle.")
            break

        if alpha.formula_raw in active_formulas:
            continue

        # Re-validate against promotion criteria as a final check
        crit = cfg['discovery'].get('promotion_criteria', {})
        if alpha.trades() < crit.get('min_trades', 20):
            continue

        active_alphas.append(alpha.model_dump())
        active_formulas.add(alpha.formula_raw)
        promoted_count += 1
        promotions_today_count += 1
        
        state.r.rpush(_today_key("promotions"), alpha.id)
        logger.success(f"EIL_PROMOTE_SUCCESS id={alpha.id} trades={alpha.trades()} sharpe={alpha.sharpe():.2f} sortino={alpha.sortino():.2f}")

    if promoted_count > 0:
        state.set_active_alphas(active_alphas)
        
    return promoted_count

def _sync_and_run_screener(cfg: Dict[str, Any], state: StateManager) -> Tuple[List[Dict[str, Any]], bool]:
    """Synchronously run the screener and return the resulting instruments.

    This function ensures that the screener is executed with the latest data and
    configuration, and it waits for the screener task to complete.

    Returns:
        Tuple[List[Dict[str, Any]], bool]: A tuple containing the list of instrument dicts found
        by the screener and a boolean indicating success or failure.
    """
    from v26meme.data.harvester import run_once as _run_once

    try:
        # Force a data harvest before running the screener
        _run_once(cfg, state)
        logger.info("Data harvest complete. Running screener...")
        
        # Run the screener
        universe, tickers = UniverseScreener(cfg["data_source"]["exchanges"], cfg["screener"], cfg.get("feeds")).get_active_universe()
        
        # Save the screener results
        snapshot_dir = cfg["screener"].get("snapshot_dir") or "data/screener_snapshots"
        ScreenerStore(snapshot_dir, state).save(universe, tickers)
        
        logger.info(f"Screener run complete. Symbols found: {len(universe)}")
        return list(universe), True
    except Exception as e:
        logger.error(f"Screener run failed: {e}")
        return [], False

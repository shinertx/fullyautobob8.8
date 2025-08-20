# v26meme/cli.py — v4.7.5 (event‑sourced data plane, canonical joins, calibrated sim, lanes)
from __future__ import annotations
import os, time, json, hashlib, random, inspect
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple
from collections.abc import Awaitable as _Awaitable  # ensure consistent with harvester _sync

import click, yaml
from dotenv import load_dotenv
from loguru import logger
import pandas as pd
import numpy as np

from v26meme.core.state import StateManager
from v26meme.data.lakehouse import Lakehouse
from v26meme.data.universe_screener import UniverseScreener
from v26meme.data.screener_store import ScreenerStore
from v26meme.data.top_gainers import compute_top_gainers_bases
from v26meme.research.feature_factory import FeatureFactory
from v26meme.research.generator import GeneticGenerator
from v26meme.research.validation import panel_cv_stats, benjamini_hochberg, deflated_sharpe_ratio
from v26meme.research.feature_prober import FeatureProber
from v26meme.labs.simlab import SimLab
from v26meme.allocation.optimizer import PortfolioOptimizer
from v26meme.allocation.lanes import LaneAllocationManager
from v26meme.execution.exchange import ExchangeFactory
from v26meme.execution.handler import ExecutionHandler
from v26meme.execution.risk import RiskManager
from v26meme.core.dsl import Alpha
from v26meme.llm.proposer import LLMProposer
from v26meme.analytics.adaptive import publish_adaptive_knobs
from v26meme.registry.resolver import configure as configure_resolver
from v26meme.registry.catalog import CatalogManager
from v26meme.data.harvester import run_once as harvest_once  # event-sourced incremental OHLCV harvest

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
    alphas = state.get_active_alphas()
    if not alphas:
        return changed
    seen = set(); clean: List[Dict[str, Any]] = []
    crit = (cfg.get('discovery') or {}).get('promotion_criteria', {})
    enforce = bool((cfg.get('discovery') or {}).get('enforce_current_gates_on_start', False))
    max_trim = int((cfg.get('discovery') or {}).get('max_return_padding_trim', 5))
    for a in alphas:
        aid = a.get('id');
        if not aid:
            continue
        if aid in seen:
            changed["dupes_removed"] += 1
            continue
        seen.add(aid)
        
        # Get performance dict, creating nested structure if needed
        perf = a.get('performance', {}).get('all', {})
        returns = list(perf.get('returns') or [])
        n_trades = int(perf.get('n_trades') or 0)
        
        # Trim condition: only if extra section exists AND consisting solely of zeros
        if n_trades > 0 and len(returns) > n_trades:
            extra = returns[n_trades:]
            if all((r or 0) == 0 for r in extra) and len(extra) <= max_trim:
                # Trim to n_trades and properly update the nested structure
                trimmed_returns = returns[:n_trades]
                # Create new perf dict with trimmed returns
                new_perf = dict(perf)
                new_perf['returns'] = trimmed_returns
                new_perf['n_trades'] = n_trades  # ensure it's set
                # Update the alpha dict properly
                a = dict(a)  # shallow copy
                a['performance'] = {'all': new_perf}
                if 'performance' in a and isinstance(a['performance'], dict):
                    for k in a['performance']:
                        if k != 'all':
                            a['performance'][k] = a['performance'][k]
                changed['trimmed'] += 1
        
        # Gate enforcement
        if enforce:
            # Re-fetch perf in case it was modified
            perf = a.get('performance', {}).get('all', {})
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
    if clean != alphas:
        state.set_active_alphas(clean)
    changed['final'] = len(clean)
    return changed

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def load_config(file: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load YAML config.
    PIT note: Config carries only lag/embargo parameters; no forward info embedded.
    """
    with open(file, "r") as f:  # noqa: P103
        return yaml.safe_load(f)

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
    min_cov = float(cfg.get('harvester', {}).get('min_coverage_for_research', 0.0))
    if state.get('coverage:raise_threshold'):
        hi = cfg.get('harvester', {}).get('high_coverage_threshold')
        if hi is not None:
            try: min_cov = float(hi)
            except Exception: pass
    min_sym = int(cfg.get('discovery', {}).get('min_panel_symbols', 1))
    min_bars = int(cfg.get('discovery', {}).get('min_bars_per_symbol', 1))
    try:
        from typing import cast, List as _List
        keys = cast(_List[str], state.r.hkeys('harvest:coverage') or [])
    except Exception:
        return False, {"error": "coverage_hash_unavailable"}
    covered: Dict[str, dict] = {}
    for k in keys:
        try:
            if isinstance(k, bytes):  # defensive
                k = k.decode('utf-8')
            parts = str(k).split(':', 2)
            if len(parts) != 3:
                continue
            _ex_id, tf_key, canonical = parts
            if tf_key != tf:
                continue
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
            if isinstance(meta, str):  # double encoded case
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
            if not prev or int(meta.get('actual') or 0) > int(prev.get('actual') or 0):
                covered[canonical] = meta
        except Exception:
            continue
    elig = [c for c, m in covered.items() if (m.get('coverage', 0) >= min_cov and (m.get('actual') or 0) >= min_bars)]
    ok = len(elig) >= min_sym
    return ok, {"eligible": len(elig), "required": min_sym, "min_cov": min_cov, "tf": tf, "symbols": len(covered)}

# --------------------------------------------------------------------------------------
# CLI Root
# --------------------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """Root CLI group."""
    pass

# --------------------------------------------------------------------------------------
# Main Loop
# --------------------------------------------------------------------------------------

@cli.command()
def loop() -> None:
    """Run the research + promotion loop (paper only).

    Safeguards:
    - Deterministic: fixed seed unless overridden in config.
    - PIT correctness: all feature transforms are lagged (see lagged_zscore).
    - No live trading side‑effects (execution handler stays in paper mode).
    """
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
    logger.info("🚀 v26meme v4.7.5 loop starting…")

    if ((cfg.get("llm") or {}).get("provider", "").lower() != "openai" or not os.getenv("OPENAI_API_KEY", "")):
        logger.warning("LLM is OpenAI-only. Set llm.provider=openai and OPENAI_API_KEY in .env.")

    state = StateManager(cfg["system"]["redis_host"], cfg["system"]["redis_port"])
    _ensure_lakehouse_bootstrap(cfg, state)

    lakehouse = _make_lakehouse(cfg)
    screener = UniverseScreener(cfg["data_source"]["exchanges"], cfg["screener"], cfg.get("feeds"))
    store = ScreenerStore(cfg["screener"].get("snapshot_dir", "data/screener_snapshots"), state)
    feature_factory = FeatureFactory()
    catalog = CatalogManager(state, cfg.get("registry"))

    slip_table = state.get("slippage:table") or {}
    try:
        simlab = SimLab(
            cfg["execution"]["paper_fees_bps"],
            cfg["execution"]["paper_slippage_bps"],
            slippage_table=slip_table,
        )
    except TypeError:  # backward compat
        simlab = SimLab(
            cfg["execution"]["paper_fees_bps"],
            cfg["execution"]["paper_slippage_bps"],
        )

    base_features = [
        "return_1p",
        "volatility_20p",
        "momentum_10p",
        "rsi_14",
        "close_vs_sma50",
        "hod_sin",
        "hod_cos",
        "round_proximity",
        "btc_corr_20p",
        "eth_btc_ratio",
    ]
    generator = GeneticGenerator(
        base_features,
        cfg["discovery"]["population_size"],
        seed=cfg["system"].get("seed", 1337),
    )
    prober = FeatureProber(
        cfg["execution"]["paper_fees_bps"],
        cfg["execution"]["paper_slippage_bps"],
        perturbations=cfg["prober"].get("perturbations", 64),
        delta_fraction=cfg["prober"].get("delta_fraction", 0.15),
        seed=cfg["system"].get("seed", 1337),
    )
    optimizer = PortfolioOptimizer(cfg)
    lane_mgr = LaneAllocationManager(cfg, state)
    exchange_factory = ExchangeFactory(os.environ.get("GCP_PROJECT_ID"))
    risk = RiskManager(state, cfg)
    exec_handler = ExecutionHandler(state, exchange_factory, cfg, risk_manager=risk)
    proposer = LLMProposer(state)

    def _research_tf() -> str:
        tf_opts = (cfg.get("harvester") or {}).get("timeframes_by_lane", {}).get("core") or ["1h"]
        return "1h" if "1h" in tf_opts else tf_opts[0]

    while True:  # Main cycle
        try:
            state.heartbeat()
            logger.info("--- New loop cycle ---")
            # Hygiene pass early each cycle (pre-harvest to keep list small during scoring)
            try:
                hg = _alpha_registry_hygiene(state, cfg)
                if sum([v for k,v in hg.items() if k != 'final']) > 0:
                    logger.info(f"[hygiene] {json.dumps(hg)}")
            except Exception as _e:
                logger.warning(f"Hygiene error: {_e}")
            catalog.maybe_refresh(
                screener.exchanges,
                include_derivatives=cfg["screener"].get("derivatives_enabled", False),
            )
            # Harvest latest OHLCV (partial mode if enabled)
            try:
                harvest_once(cfg, state, partial_mode=cfg.get('harvester', {}).get('partial_harvest', False))  # type: ignore[arg-type]
            except TypeError:
                # backward compatibility with earlier signature
                harvest_once(cfg, state)  # type: ignore[misc]
            except Exception as e:  # fail closed but keep loop alive
                logger.opt(exception=True).error(f"harvest cycle error: {e}")
                risk.note_error()

            # Coverage gate (skip research work until minimal panel viable)
            tf_gate = _research_tf()
            gate_ok, gate_stats = _coverage_gate_ok(state, cfg, tf_gate)
            if not gate_ok:
                logger.info(f"Coverage gate pending tf={gate_stats.get('tf')} eligible={gate_stats.get('eligible')}/{gate_stats.get('required')} min_cov={gate_stats.get('min_cov')}")
                time.sleep(cfg["system"]["loop_interval_seconds"])
                continue

            instruments, tickers_by_venue = screener.get_active_universe()
            if not instruments:
                logger.warning("No instruments from screener; sleeping.")
                time.sleep(cfg["system"]["loop_interval_seconds"])
                continue

            # Save screener snapshot + publish latest symbols & compute top gainers
            store.save(instruments, tickers_by_venue)
            moonshot_bases = compute_top_gainers_bases(tickers_by_venue, top_n=10)
            state.set("lane:moonshot:gainers", list(moonshot_bases))

            # Adaptive knobs (vol‑scaled daily stop)
            tf = _research_tf()
            btc_df = _lh_get_data(lakehouse, cfg, "BTC_USD_SPOT", tf)
            publish_adaptive_knobs(state, cfg, btc_df)

            adapt_pop = state.get("adaptive:population_size")
            if adapt_pop:
                generator.population_size = int(adapt_pop)

            lh_syms = set(_lh_get_available_symbols(lakehouse, cfg, tf))
            tradeable: List[Tuple[Dict[str, Any], str]] = []
            for inst in instruments:
                canon = inst.get("display")
                if canon in lh_syms:
                    tradeable.append((inst, canon))
            if not tradeable:
                logger.warning("No tradeable symbols intersect lakehouse; sleeping.")
                time.sleep(cfg["system"]["loop_interval_seconds"])
                continue
            logger.info(f"Tradeable: {[c for _, c in tradeable]}")

            if not generator.population:
                generator.initialize_population()
            # Population pruning (memory / search pressure control)
            try:
                max_pop = int(cfg.get('discovery', {}).get('max_population_size', 1000))
                if len(generator.population) > max_pop:
                    generator.population = generator.population[:max_pop]
            except Exception:
                pass

            # Ingest EIL survivors (budget)
            if cfg.get("eil", {}).get("enabled", True):
                scan_count = int(cfg.get('eil', {}).get('scan_batch_size', 200))
                keys = [k for k in state.r.scan_iter(match="eil:candidates:*", count=scan_count)]
                keys = keys[: int(cfg["eil"].get("survivor_top_k", 25))]
                for k in keys:
                    cand = state.get(k)
                    if cand and cand.get("formula"):
                        generator.population.append(cand["formula"])
                for k in keys:
                    state.r.delete(k)

            # LLM suggestions (OpenAI-only)
            if cfg.get("llm", {}).get("enable", True):
                k = int(cfg["llm"].get("max_suggestions_per_cycle", 3))
                for f in proposer.propose(base_features, k=k):
                    if f not in generator.population:
                        generator.population.append(f)

            # Build panel cache
            eth_df = _lh_get_data(lakehouse, cfg, "ETH_USD_SPOT", tf)
            df_cache: Dict[str, pd.DataFrame] = {}
            bases = list({canon.split('_')[0] for _, canon in tradeable})
            if not bases:
                logger.warning("No bases extracted from tradeable list; sleeping.")
                time.sleep(cfg["system"]["loop_interval_seconds"])
                continue
            k_bases = min(max(1, int(cfg["discovery"]["panel_symbols"])), len(bases))
            chosen_bases = random.sample(bases, k_bases)
            for base in chosen_bases:
                canon = f"{base}_USD_SPOT"
                df = _lh_get_data(lakehouse, cfg, canon, tf)
                if df.empty:
                    continue
                df_feat = feature_factory.create(
                    df,
                    symbol=canon,
                    cfg=cfg,
                    other_dfs={"BTC_USD_SPOT": btc_df, "ETH_USD_SPOT": eth_df},
                )
                for f in base_features:
                    if f in df_feat.columns:
                        lookback = cfg.get('features', {}).get('zscore_lookback', 200)
                        df_feat[f] = lagged_zscore(df_feat[f], lookback=lookback, cfg=cfg)
                df_feat = df_feat.dropna()
                df_feat.attrs["display"] = canon
                df_cache[canon] = df_feat

            # Fitness via panel CV stats
            cv_method = str(cfg["discovery"].get("cv_method", "kfold")).lower()
            pvals_by_fid: Dict[str, float] = {}
            fitness: Dict[str, float] = {}
            for formula in list(generator.population):  # snapshot
                fid = hashlib.sha256(json.dumps(formula).encode()).hexdigest()
                panel_returns: Dict[str, pd.Series] = {}
                for base in chosen_bases:
                    canon = f"{base}_USD_SPOT"
                    dff = df_cache.get(canon)
                    if dff is None or dff.empty:
                        continue
                    stats = simlab.run_backtest(dff, formula)
                    overall = stats.get("all", {})
                    if overall and overall.get("n_trades", 0) > 0:
                        panel_returns[canon] = pd.Series(overall.get("returns", []), dtype=float)
                if not panel_returns:
                    fitness[fid] = 0.0
                    continue
                cv = panel_cv_stats(
                    panel_returns,
                    k_folds=cfg["discovery"]["cv_folds"],
                    embargo=cfg["discovery"]["cv_embargo_bars"],
                    alpha_fdr=cfg["discovery"]["fdr_alpha"],
                    cv_method=cv_method,
                )
                fitness[fid] = float(cv.get("mean_oos", 0.0))
                pvals_by_fid[fid] = float(cv.get("p_value", 1.0))

            # Evolve population using available fitness scores
            generator.run_evolution_cycle(fitness)

            # Promotion pass: BH-FDR + hard gates + factor-aware penalty
            rep_canon = next(iter(df_cache.keys()), None)
            promoted_candidates: List[Dict[str, Any]] = []
            if rep_canon:
                # Build p-value pairs from stored stats when available
                pairs = []
                for fid, pv in pvals_by_fid.items():
                    # allow external stats override
                    stats = state.get(f"alpha:stats:{fid}") or {}
                    real_p = stats.get('p_value')
                    if real_p is not None:
                        pv = float(real_p)
                    pairs.append((fid, float(pv)))
                if pairs:
                    pvals = [p for _, p in pairs]
                    kept_mask, _ = benjamini_hochberg(pvals, cfg["discovery"]["fdr_alpha"])
                    keep_fids = {fid for (fid, _), keep in zip(pairs, kept_mask) if keep}
                else:
                    keep_fids = set()
                prom: List[Dict[str, Any]] = []
                df_rep = df_cache.get(rep_canon)
                for formula in generator.population:
                    fid = hashlib.sha256(json.dumps(formula).encode()).hexdigest()
                    if fid not in keep_fids or df_rep is None:
                        continue
                    stats_rep = simlab.run_backtest(df_rep, formula)
                    overall = stats_rep.get("all", {})
                    if not overall or overall.get("n_trades", 0) == 0:
                        continue
                    pv = pvals_by_fid.get(fid, 1.0)

                    # factor penalty vs current active alphas
                    active = state.get_active_alphas()
                    corr_pen = 0.0
                    if active:
                        cand = np.array(overall.get("returns", []), dtype=float)
                        for a in active:
                            ar = np.array((a.get("performance", {}).get("all", {}).get("returns", [])), dtype=float)
                            if len(ar) > 5 and len(cand) > 5:
                                m = min(len(ar), len(cand))
                                c = np.corrcoef(ar[-m:], cand[-m:])[0, 1]
                                if np.isfinite(c):
                                    corr_pen = max(corr_pen, abs(float(c)))
                    if corr_pen >= cfg["discovery"]["factor_correlation_max"]:
                        pv = min(1.0, pv + 0.4)  # penalize crowding
                    prom.append({"fid": fid, "formula": formula, "p": pv, "overall": overall})
                if prom:
                    pvals = [c["p"] for c in prom]
                    kept_mask, _ = benjamini_hochberg(pvals, cfg["discovery"]["fdr_alpha"])
                    promoted_candidates = [c for c, keep in zip(prom, kept_mask) if keep]

            # Promotion Gate 2: Hard gates + prober + lane tag
            budget_left = cfg["discovery"]["max_promotions_per_cycle"]
            gate = cfg["discovery"]["promotion_criteria"]
            min_rob = float(cfg["prober"].get("min_robust_score", 0.55))
            moon_base_set = set(state.get("lane:moonshot:gainers") or [])
            dsr_cfg = (cfg.get("validation") or {}).get("dsr", {})
            # Rejection counters (1):
            rej = {"dsr":0, "min_trades":0, "hard_gates":0, "factor_corr":0, "robust":0}
            daily_key = f"promotions:day:{datetime.now(timezone.utc).strftime('%Y%m%d')}"
            used_today = int(state.get(daily_key) or 0)
            daily_cap = int(cfg['discovery'].get('max_promotions_per_day', 0))
            if used_today >= daily_cap:
                logger.info(f"Daily promotion cap reached ({used_today}/{daily_cap})")
                promoted_candidates = []
            # Risk freeze (E): if conserve_mode active or halted, block promotions
            if state.get('risk:halted') or state.get('risk:conserve_active'):
                if promoted_candidates:
                    logger.info("Promotion freeze active (risk conserve/halt)")
                promoted_candidates = []
            for c in sorted(promoted_candidates, key=lambda x: x["overall"]["sortino"], reverse=True):
                if budget_left <= 0:
                    break
                if used_today >= daily_cap:
                    break
                ov = c["overall"]
                fid = c["fid"]
                if dsr_cfg.get("enabled"):
                    n_trials = max(1, len(generator.population))
                    dsr_prob = deflated_sharpe_ratio(pd.Series(ov.get("returns", []), dtype=float), n_trials=n_trials, sr_benchmark=float(dsr_cfg.get("benchmark_sr", 0.0)))
                    if dsr_prob < float(dsr_cfg.get("min_prob", 0.60)):
                        rej['dsr'] += 1; continue
                buf = float(cfg.get('discovery', {}).get('promotion_buffer_multiplier', 1.0))
                if ov.get("n_trades", 0) < gate["min_trades"]:
                    rej['min_trades'] += 1; continue
                if not (ov.get("sortino", 0) >= gate["min_sortino"] * buf and ov.get("sharpe", 0) >= gate["min_sharpe"] * buf and ov.get("win_rate", 0) >= gate.get("min_win_rate", 0) and ov.get("mdd", 0) <= gate.get("max_mdd", 1.0)):
                    rej['hard_gates'] += 1; continue
                rob_score = 0.0
                if cfg.get("prober", {}).get("enabled", True):
                    if rep_canon and df_cache.get(rep_canon) is not None:
                        probe_res = prober.score(df_cache[rep_canon], c["formula"])
                        rob_score = float(probe_res.get("robust_score", 0.0))
                if rob_score < min_rob:
                    rej['robust'] += 1; continue
                # Factor correlation penalty already applied via pv earlier; enforce hard cap
                active = state.get_active_alphas()
                crowd = False
                if active:
                    cand = np.array(ov.get("returns", []), dtype=float)
                    for a in active:
                        ar = np.array((a.get("performance", {}).get("all", {}).get("returns", [])), dtype=float)
                        if len(ar) > 5 and len(cand) > 5:
                            m = min(len(ar), len(cand))
                            cval = np.corrcoef(ar[-m:], cand[-m:])[0, 1]
                            if np.isfinite(cval) and abs(float(cval)) >= cfg["discovery"]["factor_correlation_max"]:
                                crowd = True; break
                if crowd:
                    rej['factor_corr'] += 1; continue
                alpha_id = c["fid"]
                alpha = Alpha(
                    id=alpha_id,
                    name=alpha_id[:8],
                    formula=c["formula"],
                    universe=[rep_canon] if rep_canon else [],
                    timeframe=_research_tf(),
                    lane="moonshot" if (rep_canon.split("_")[0] if rep_canon else "") in moon_base_set else "core",
                    performance={"all": ov},
                )
                # Atomic-ish update with dedupe safeguard
                actives = state.get_active_alphas(); actives.append(alpha.model_dump())
                seen_ids: set[str] = set(); deduped: list[dict[str, Any]] = []
                for _a in actives:
                    aid = _a.get('id')
                    if not aid or aid in seen_ids: continue
                    seen_ids.add(aid); deduped.append(_a)
                state.set_active_alphas(deduped)
                # Flag for coverage threshold escalation after first promotion
                if not state.get('coverage:raise_threshold'):
                    state.set('coverage:raise_threshold', 1)
                budget_left -= 1; used_today += 1
                state.set(daily_key, used_today)
                logger.success(
                    f"PROMOTED {alpha.id[:8]} lane={alpha.lane} rob={rob_score:.2f} sharpe={ov.get('sharpe',0):.2f} trades={ov.get('n_trades')} (used_today={used_today}/{daily_cap})"
                )
            if rej:
                logger.info(f"Promotion rejections breakdown {rej} (remaining_daily={daily_cap-used_today})")
            # Snapshot active alphas registry (audit) (3)
            try:
                snap_dir = Path('data/alphas'); snap_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
                import json as _json
                with open(snap_dir / f'registry_{ts}.json','w') as f:
                    _json.dump(state.get_active_alphas(), f)
            except Exception:
                logger.warning("Could not write alpha registry snapshot")

            # Auto-switch to CPCV if configured (7)
            try:
                if cfg['discovery'].get('auto_cpcv'):
                    if (cfg['discovery']['cv_method'] == 'kfold' and (len(df_cache) >= 8 or cfg['discovery']['population_size'] >= 300)):
                        cfg['discovery']['cv_method'] = 'cpcv'
                        logger.info("Auto-switched CV method to CPCV for robustness")
            except Exception:
                pass

            try:
                active_alphas = state.get_active_alphas()
                if active_alphas:
                    raw_weights = optimizer.get_weights(active_alphas, 'all')
                    if raw_weights:
                        # Lane snapshot
                        try:
                            lane_snapshot = {a['id'][:8]: a.get('lane','core') for a in active_alphas}
                            logger.info(f"Lane snapshot active={len(active_alphas)} lanes={lane_snapshot}")
                        except Exception:
                            pass
                        lane_weights = lane_mgr.apply_lane_budgets(raw_weights, active_alphas)
                        state.set('portfolio:alpha_weights', lane_weights)
                        exec_handler.reconcile(lane_weights, active_alphas)
                        logger.info(f"Reconciled portfolio with {len(lane_weights)} alpha weights post-lane allocation")
                    else:
                        logger.info("No optimizer weights produced (insufficient performance data)")
                else:
                    logger.info("No active alphas to allocate this cycle")
            except Exception as e:
                logger.opt(exception=True).error(f"Allocation/reconcile error: {e}")

        except KeyboardInterrupt:
            logger.warning("Shutdown requested.")
            break
        except Exception as e:  # fail-closed with backoff
            logger.opt(exception=True).error(f"Loop error: {e}")
            risk.note_error()
            time.sleep(cfg["system"]["loop_interval_seconds"] * 2)

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
    instruments, tickers_by_venue = screener.get_active_universe()
    print("Instruments:", instruments)
    print("Tickers by venue:", list(tickers_by_venue.keys()))

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

if __name__ == "__main__":  # pragma: no cover
    cli()

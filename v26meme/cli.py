# v26meme/cli.py — v4.7.5 (event‑sourced data plane, canonical joins, calibrated sim, lanes)
import click, yaml, os, time, json, hashlib, random, inspect
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from v26meme.core.state import StateManage            logger.logger.opt(exception=True).error(f"Loop error: {e}")
            if risk: risk.note_error()
            time.sleep(cfg["system"]["loop_interval_seconds"]*2)

@click.command()
def debug_screener():m v26meme.data.lakehouse import Lakehouse
from v26meme.data.universe_screener import UniverseScreener
from v26meme.data.screener_store import ScreenerStore
from v26meme.data.top_gainers import compute_top_gainers_bases
from v26meme.research.feature_factory import FeatureFactory
from v26meme.research.generator import GeneticGenerator
from v26meme.research.validation import panel_cv_stats, benjamini_hochberg
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

def load_config(file: str = "configs/config.yaml"):
    with open(file, "r") as f: return yaml.safe_load(f)

def lagged_zscore(s: pd.Series, lookback: int = 200) -> pd.Series:
    m = s.rolling(lookback, min_periods=max(5, lookback//4)).mean().shift(1)
    sd = s.rolling(lookback, min_periods=max(5, lookback//4)).std().shift(1)
    return (s - m) / sd

def _ensure_lakehouse_bootstrap(cfg: dict, state: StateManager):
    """Bootstrap the lakehouse with an initial harvest if empty.

    Uses the event-sourced harvester's run_once function (paper-safe). State is
    passed explicitly so checkpoints are recorded correctly.
    """
    root = Path("data")
    if any(root.glob("**/*.parquet")):
        return
    logger.info("No lakehouse detected; harvesting initial dataset (first run)…")
    from v26meme.data.harvester import run_once as _run_once
    _run_once(cfg, state)
    logger.info("Initial harvest complete.")

def _make_lakehouse(cfg):
    try:
        if "preferred_exchange" in inspect.signature(Lakehouse).parameters:
            return Lakehouse(preferred_exchange=cfg["execution"]["primary_exchange"])
    except Exception:
        pass
    return Lakehouse()

def _lh_get_data(lh, cfg, canonical: str, tf: str) -> pd.DataFrame:
    try: return lh.get_data(canonical, tf)
    except TypeError:
        try: return lh.get_data(cfg["execution"]["primary_exchange"], canonical, tf)
        except Exception: return pd.DataFrame()

def _lh_get_available_symbols(lh, cfg, tf: str) -> list:
    try: return lh.get_available_symbols(tf)
    except TypeError:
        try: return lh.get_available_symbols(cfg["execution"]["primary_exchange"], tf)
        except Exception: return []

def _today_key(prefix="promotions"):  # YYYY-MM-DD key
    return f"{prefix}:{datetime.now(timezone.utc).date().isoformat()}"

@click.group()
def cli(): pass

@cli.command()
def loop():
#     load_dotenv()
#     cfg = load_config()
#     random.seed(cfg["system"].get("seed", 1337))
#     configure_resolver(cfg.get("registry"))
# 
#     Path("logs").mkdir(exist_ok=True, parents=True)
#     logger.add("logs/system.log", level=cfg["system"]["log_level"], rotation="10 MB",
#                retention="14 days", enqueue=True)
#     logger.info("🚀 v26meme v4.7.5 loop starting…")
# 
#     if ((cfg.get("llm") or {}).get("provider","").lower() != "openai" or not os.getenv("OPENAI_API_KEY","")):
#         logger.warning("LLM is OpenAI-only. Set llm.provider=openai and OPENAI_API_KEY in .env.")
# 
#     state = StateManager(cfg["system"]["redis_host"], cfg["system"]["redis_port"])
#     _ensure_lakehouse_bootstrap(cfg, state)
# 
#     lakehouse = _make_lakehouse(cfg)
#     screener = UniverseScreener(cfg["data_source"]["exchanges"], cfg["screener"], cfg.get("feeds"))
#     store = ScreenerStore(cfg["screener"].get("snapshot_dir", "data/screener_snapshots"), state)
#     feature_factory = FeatureFactory()
#     catalog = CatalogManager(state, cfg.get("registry"))
# 
#     slip_table = state.get("slippage:table") or {}
#     try:
#         simlab = SimLab(cfg["execution"]["paper_fees_bps"], cfg["execution"]["paper_slippage_bps"],
#                         slippage_table=slip_table)
#     except TypeError:
#         simlab = SimLab(cfg["execution"]["paper_fees_bps"], cfg["execution"]["paper_slippage_bps"])
# 
#     base_features = ["return_1p","volatility_20p","momentum_10p","rsi_14","close_vs_sma50",
#                      "hod_sin","hod_cos","round_proximity","btc_corr_20p","eth_btc_ratio"]
#     generator = GeneticGenerator(base_features, cfg["discovery"]["population_size"], seed=cfg["system"].get("seed",1337))
#     prober = FeatureProber(cfg["execution"]["paper_fees_bps"], cfg["execution"]["paper_slippage_bps"],
#                            perturbations=cfg["prober"].get("perturbations",64),
#                            delta_fraction=cfg["prober"].get("delta_fraction",0.15),
#                            seed=cfg["system"].get("seed",1337))
#     optimizer = PortfolioOptimizer(cfg)
#     lane_mgr = LaneAllocationManager(cfg, state)
#     exchange_factory = ExchangeFactory(os.environ.get("GCP_PROJECT_ID"))
#     risk = RiskManager(state, cfg)
#     exec_handler = ExecutionHandler(state, exchange_factory, cfg, risk_manager=risk)
#     proposer = LLMProposer(state)
# 
#     def _research_tf():
#         tf_opts = (cfg.get("harvester") or {}).get("timeframes_by_lane", {}).get("core") or ["1h"]
#         return "1h" if "1h" in tf_opts else tf_opts[0]
# 
#     while True:
#         try:
#             state.heartbeat()
#             logger.info("--- New loop cycle ---")
#             catalog.maybe_refresh(screener.exchanges, include_derivatives=cfg["screener"].get("derivatives_enabled", False))
# 
#             # Harvest latest OHLCV bars (event-sourced, resumable, QA-gated)
#             try:
#                 harvest_once(cfg, state)
#             except Exception as e:  # fail closed but keep loop alive
#                 logger.opt(exception=True).error(f"harvest cycle error: {e}")
#                 if risk: risk.note_error()
# 
#             instruments, tickers_by_venue = screener.get_active_universe()
#             if not instruments:
#                 logger.warning("No instruments from screener; sleeping.")
#                 time.sleep(cfg["system"]["loop_interval_seconds"]); continue
# 
#             # Save screener snapshot + publish latest symbols & compute top gainers
#             store.save(instruments, tickers_by_venue)
#             moonshot_bases = compute_top_gainers_bases(tickers_by_venue, top_n=10)
#             state.set("lane:moonshot:gainers", list(moonshot_bases))
# 
#             # Adaptive knobs (vol‑scaled daily stop)
#             tf = _research_tf()
#             btc_df = _lh_get_data(lakehouse, cfg, "BTC_USD_SPOT", tf)
#             publish_adaptive_knobs(state, cfg, btc_df)
# 
#             adapt_pop = state.get("adaptive:population_size")
#             if adapt_pop: generator.population_size = int(adapt_pop)
# 
#             lh_syms = set(_lh_get_available_symbols(lakehouse, cfg, tf))
#             tradeable = []
#             for inst in instruments:
#                 canon = inst.get("display")
#                 if canon in lh_syms: tradeable.append((inst, canon))
#             if not tradeable:
#                 logger.warning("No tradeable symbols intersect lakehouse; sleeping.")
#                 time.sleep(cfg["system"]["loop_interval_seconds"]); continue
#             logger.info(f"Tradeable: {[c for _,c in tradeable]}")
# 
#             if not generator.population: generator.initialize_population()
# 
#             # Ingest EIL survivors (budget)
#             if cfg.get("eil",{}).get("enabled", True):
#                 keys = [k for k in state.r.scan_iter(match="eil:candidates:*", count=200)]
#                 keys = keys[: int(cfg["eil"].get("survivor_top_k", 25))]
#                 for k in keys:
#                     cand = state.get(k)
#                     if cand and cand.get("formula"):
#                         generator.population.append(cand["formula"])
#                 for k in keys: state.r.delete(k)
# 
#             # LLM suggestions (OpenAI-only)
#             if cfg.get("llm",{}).get("enable", True):
#                 k = int(cfg["llm"].get("max_suggestions_per_cycle",3))
#                 for f in proposer.propose(base_features, k=k):
#                     if f not in generator.population: generator.population.append(f)
# 
#             # Build panel cache
#             eth_df = _lh_get_data(lakehouse, cfg, "ETH_USD_SPOT", tf)
#             df_cache = {}
#             bases = list({canon.split('_')[0] for _, canon in tradeable})
#             chosen_bases = random.sample(bases, min(max(1,cfg["discovery"]["panel_symbols"])), len(bases)))
#             for base in chosen_bases:
#                 canon = f"{base}_USD_SPOT"
#                 df = _lh_get_data(lakehouse, cfg, canon, tf)
#                 if df.empty: continue
#                 df_feat = feature_factory.create(df, symbol=canon, cfg=cfg,
#                                                  other_dfs={'BTC_USD_SPOT': btc_df, 'ETH_USD_SPOT': eth_df})
#                 for f in base_features:
#                     df_feat[f] = lagged_zscore(df_feat[f], lookback=200)
#                 df_feat = df_feat.dropna()
#                 df_feat.attrs["display"] = canon
#                 df_cache[canon] = df_feat
# 
#             # Fitness via panel CV stats
#             cv_method = str(cfg["discovery"].get("cv_method", "kfold")).lower()
#             pvals_by_fid = {}
#             fitness = {}
#             for formula in generator.population:
#                 fid = hashlib.sha256(json.dumps(formula).encode()).hexdigest()
#                 panel_returns = {}
#                 for base in chosen_bases:
#                     canon = f"{base}_USD_SPOT"
#                     dff = df_cache.get(canon)
#                     if dff is None or dff.empty: continue
#                     stats = simlab.run_backtest(dff, formula)
#                     overall = stats.get('all', {})
#                     if overall and overall.get('n_trades', 0) > 0:
#                         panel_returns[canon] = pd.Series(overall.get('returns', []), dtype=float)
#                 if not panel_returns:
#                     fitness[fid] = 0.0
#                     continue
#                 cv = panel_cv_stats(panel_returns,
#                                     k_folds=cfg['discovery']['cv_folds'],
#                                     embargo=cfg['discovery']['cv_embargo_bars'],
#                                     alpha_fdr=cfg['discovery']['fdr_alpha'],
#                                     cv_method=cv_method)
#                 fitness[fid] = float(cv.get('mean_oos', 0.0))
#                 pvals_by_fid[fid] = float(cv.get('p_value', 1.0))
# 
#             generator.report_fitness(fitness)
#             generator.evolve()
# 
#             # Promotion pass: BH-FDR + hard gates + factor-aware penalty
#             rep_canon = next(iter(df_cache.keys()), None)
#             promoted_candidates = []
#             if rep_canon:
#                 df_rep = df_cache.get(rep_canon)
#                 prom = []
#                 for formula in generator.population:
#                     fid = hashlib.sha256(json.dumps(formula).encode()).hexdigest()
#                     if df_rep is None: continue
#                     stats_rep = simlab.run_backtest(df_rep, formula)
#                     overall = stats_rep.get('all', {})
#                     if not overall or overall.get('n_trades',0)==0: continue
#                     
#                     pv = pvals_by_fid.get(fid, 1.0)
# 
#                     # factor penalty vs current active alphas
#                     active = state.get_active_alphas()
#                     corr_pen = 0.0
#                     if active:
#                         cand = np.array(overall.get('returns', []), dtype=float)
#                         for a in active:
#                             ar = np.array((a.get('performance',{}).get('all',{}).get('returns',[])), dtype=float)
#                             if len(ar)>5 and len(cand)>5:
#                                 m = min(len(ar), len(cand))
#                                 c = np.corrcoef(ar[-m:], cand[-m:])[0,1]
#                                 if np.isfinite(c): corr_pen = max(corr_pen, abs(float(c)))
#                     if corr_pen >= cfg['discovery']['factor_correlation_max']:
#                         # Penalize p-value heavily to avoid crowding
#                         pv = min(1.0, pv + 0.4)
#                     prom.append({"fid": fid, "formula": formula, "p": pv, "overall": overall})
#                 if prom:
#                     pvals = [c["p"] for c in prom]
#                     kept_mask, _ = benjamini_hochberg(pvals, cfg['discovery']['fdr_alpha'])
#                     promoted_candidates = [c for c, keep in zip(prom, kept_mask) if keep]
# 
#             # Promotion Gate 2: Hard gates + prober + lane tag
#             budget_left = cfg['discovery']['max_promotions_per_cycle']
#             gate = cfg['discovery']['promotion_criteria']
#             min_rob = float(cfg['prober'].get('min_robust_score', 0.55))
#             moon_base_set = set(state.get("lane:moonshot:gainers") or [])
#             for c in sorted(promoted_candidates, key=lambda x: x['overall']['sortino'], reverse=True):
#                 if budget_left <= 0: break
#                 ov = c['overall']
#                 
#                 # Deflated Sharpe Ratio gate (if enabled)
#                 fid = c["fid"]; dsr_conf_ok = True
#                 dsr_cfg = (cfg.get("validation") or {}).get("dsr", {})
#                 if dsr_cfg.get("enabled"):
#                     p_raw = pvals_by_fid.get(fid, 1.0)
#                     n_strats = len(generator.population) or 1
#                     false_pos_prob = min(1.0, p_raw * n_strats)
#                     prob_not_overfit = 1.0 - false_pos_prob
#                     if prob_not_overfit < float(dsr_cfg.get("min_prob", 0.95)):
#                         dsr_conf_ok = False
#                         logger.info(f"DSR gate: rejecting {fid[:6]}... (confidence {prob_not_overfit:.2f} < min_prob)")
#                 if not dsr_conf_ok:
#                     continue
# 
#                 if ov.get('n_trades',0) >= gate['min_trades'] and \
#                    ov.get('sortino',0) >= gate['min_sortino'] and \
#                    ov.get('sharpe',0)  >= gate['min_sharpe']  and \
#                    ov.get('win_rate',0) >= gate.get('min_win_rate', 0):
#                     if ov.get('mdd',0) <= gate.get('max_mdd', 1.0):
#                         # Apply prober robustness gate
#                         rob_score = 0.0
#                         if cfg.get('prober',{}).get('enabled',True):
#                             if rep_canon and df_cache.get(rep_canon) is not None:
#                                 rob_score = prober.probe(df_cache[rep_canon], c['formula'])
#                         
#                         if rob_score >= min_rob:
#                             alpha = Alpha(c['fid'], c['formula'], ov, rob_score)
#                             # Lane tagging
#                             base = rep_canon.split('_')[0]
#                             alpha.lane = "moonshot" if base in moon_base_set else "core"
#                             state.promote_alpha(alpha)
#                             budget_left -= 1
#                             logger.success(f"PROMOTED {alpha.fid[:8]} lane={alpha.lane} rob={rob_score:.2f} sharpe={ov['sharpe']:.2f} trades={ov['n_trades']}")
# 
#             # Log batch-level Probability of Backtest Overfitting (PBO)
#             try:
#                 if (cfg.get('validation', {}).get('dsr', {}).get('enabled') or True):
#                     # Compute PBO as fraction of splits with negative rank correlation
#                     symbols = [f"{b}_USD_SPOT" for b in chosen_bases]
#                     # Build performance matrix (sharpe per formula per symbol)
#                     perf_matrix: Dict[str, Dict[str, float]] = {}
#                     for formula in generator.population:
#                         fid = hashlib.sha256(json.dumps(formula).encode()).hexdigest()
#                         perf_matrix[fid] = {}
#                         for base in symbols:
#                             dff = df_cache.get(base)
#                             if dff is None or dff.empty: 
#                                 continue
#                             stats = simlab.run_backtest(dff, formula).get("all", {})
#                             sharpe = float(stats.get("sharpe", 0.0)) if stats else 0.0
#                             perf_matrix[fid][base] = sharpe
#                     n_sym = len(symbols)
#                     if n_sym >= 2 and perf_matrix:
#                         from itertools import combinations
#                         design_size = n_sym // 2
#                         comb_list = list(combinations(symbols, design_size))
#                         neg_count = 0; total = 0
#                         for design_syms in comb_list:
#                             test_syms = [s for s in symbols if s not in design_syms]
#                             design_vals = []; test_vals = []
#                             for fid, perfs in perf_matrix.items():
#                                 # average sharpe across design and test sets
#                                 d_perf = np.mean([perfs.get(sym, 0.0) for sym in design_syms])
#                                 t_perf = np.mean([perfs.get(sym, 0.0) for sym in test_syms])
#                                 design_vals.append(d_perf); test_vals.append(t_perf)
#                             # Spearman rank correlation
#                             if np.std(design_vals) == 0 or np.std(test_vals) == 0:
#                                 continue  # skip degenerate case
#                             corr = np.corrcoef(pd.Series(design_vals).rank(), pd.Series(test_vals).rank())[0,1]
#                             total += 1
#                             if corr < 0:
#                                 neg_count += 1
#                         if total > 0:
#                             pbo = neg_count / float(total)
#                             logger.info(f"PBO (Probability of Backtest Overfitting) this cycle: {pbo:.1%}")
#             except Exception as e:
#                 logger.warning(f"PBO calculation failed: {e}")
# 
#         except KeyboardInterrupt:
#             logger.warning("Shutdown requested."); break
#         except Exception as e:
#             logger.logger.opt(exception=True).error(f"Loop error: {e}")
#             if risk: risk.note_error()
#             time.sleep(cfg["system"]["loop_interval_seconds"]*2)

def debug_screener():
    load_dotenv()
    cfg = load_config()
    random.seed(cfg["system"].get("seed", 1337))
    configure_resolver(cfg.get("registry"))

    Path("logs").mkdir(exist_ok=True, parents=True)
    logger.add("logs/system.log", level=cfg["system"]["log_level"], rotation="10 MB",
               retention="14 days", enqueue=True)
    logger.info("🚀 v26meme v4.7.5 debug_screener starting…")

    screener = UniverseScreener(cfg["data_source"]["exchanges"], cfg["screener"], cfg.get("feeds"))
    instruments, tickers_by_venue = screener.get_active_universe()
    print("Instruments:", instruments)
    print("Tickers by venue:", tickers_by_venue)

if __name__ == "__main__":
    cli()

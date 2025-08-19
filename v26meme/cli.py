# v26meme/cli.py — v4.7.5 (event‑sourced data plane, canonical joins, calibrated sim, lanes)
import click, yaml, os, time, json, hashlib, random, inspect
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from v26meme.core.state import StateManager
from v26meme.data.lakehouse import Lakehouse
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
    load_dotenv()
    cfg = load_config()
    random.seed(cfg["system"].get("seed", 1337))

    Path("logs").mkdir(exist_ok=True, parents=True)
    logger.add("logs/system.log", level=cfg["system"]["log_level"], rotation="10 MB",
               retention="14 days", enqueue=True)
    logger.info("🚀 v26meme v4.7.5 loop starting…")

    if ((cfg.get("llm") or {}).get("provider","").lower() != "openai" or not os.getenv("OPENAI_API_KEY","")):
        logger.warning("LLM is OpenAI-only. Set llm.provider=openai and OPENAI_API_KEY in .env.")

    state = StateManager(cfg["system"]["redis_host"], cfg["system"]["redis_port"])
    _ensure_lakehouse_bootstrap(cfg, state)

    lakehouse = _make_lakehouse(cfg)
    screener = UniverseScreener(cfg["data_source"]["exchanges"], cfg["screener"], cfg.get("feeds"))
    store = ScreenerStore(cfg["screener"].get("snapshot_dir", "data/screener_snapshots"), state)
    feature_factory = FeatureFactory()

    slip_table = state.get("slippage:table") or {}
    try:
        simlab = SimLab(cfg["execution"]["paper_fees_bps"], cfg["execution"]["paper_slippage_bps"],
                        slippage_table=slip_table)
    except TypeError:
        simlab = SimLab(cfg["execution"]["paper_fees_bps"], cfg["execution"]["paper_slippage_bps"])

    base_features = ["return_1p","volatility_20p","momentum_10p","rsi_14","close_vs_sma50",
                     "hod_sin","hod_cos","round_proximity","btc_corr_20p","eth_btc_ratio"]
    generator = GeneticGenerator(base_features, cfg["discovery"]["population_size"], seed=cfg["system"].get("seed",1337))
    prober = FeatureProber(cfg["execution"]["paper_fees_bps"], cfg["execution"]["paper_slippage_bps"],
                           perturbations=cfg["prober"].get("perturbations",64),
                           delta_fraction=cfg["prober"].get("delta_fraction",0.15),
                           seed=cfg["system"].get("seed",1337))
    optimizer = PortfolioOptimizer(cfg)
    lane_mgr = LaneAllocationManager(cfg, state)
    exchange_factory = ExchangeFactory(os.environ.get("GCP_PROJECT_ID"))
    risk = RiskManager(state, cfg)
    exec_handler = ExecutionHandler(state, exchange_factory, cfg, risk_manager=risk)
    proposer = LLMProposer(state)

    def _research_tf():
        tf_opts = (cfg.get("harvester") or {}).get("timeframes_by_lane", {}).get("core") or ["1h"]
        return "1h" if "1h" in tf_opts else tf_opts[0]

    while True:
        try:
            state.heartbeat()
            logger.info("--- New loop cycle ---")

            instruments, tickers_by_venue = screener.get_active_universe()
            if not instruments:
                logger.warning("No instruments from screener; sleeping.")
                time.sleep(cfg["system"]["loop_interval_seconds"]); continue

            # Save screener snapshot + publish latest symbols & compute top gainers
            store.save(instruments, tickers_by_venue)
            moonshot_bases = compute_top_gainers_bases(tickers_by_venue, top_n=10)
            state.set("lane:moonshot:gainers", list(moonshot_bases))

            # Adaptive knobs (vol‑scaled daily stop)
            tf = _research_tf()
            btc_df = _lh_get_data(lakehouse, cfg, "BTC_USD_SPOT", tf)
            publish_adaptive_knobs(state, cfg, btc_df)

            adapt_pop = state.get("adaptive:population_size")
            if adapt_pop: generator.population_size = int(adapt_pop)

            lh_syms = set(_lh_get_available_symbols(lakehouse, cfg, tf))
            tradeable = []
            for inst in instruments:
                canon = inst.get("display")
                if canon in lh_syms: tradeable.append((inst, canon))
            if not tradeable:
                logger.warning("No tradeable symbols intersect lakehouse; sleeping.")
                time.sleep(cfg["system"]["loop_interval_seconds"]); continue
            logger.info(f"Tradeable: {[c for _,c in tradeable]}")

            if not generator.population: generator.initialize_population()

            # Ingest EIL survivors (budget)
            if cfg.get("eil",{}).get("enabled", True):
                keys = [k for k in state.r.scan_iter(match="eil:candidates:*", count=200)]
                keys = keys[: int(cfg["eil"].get("survivor_top_k", 25))]
                for k in keys:
                    cand = state.get(k)
                    if cand and cand.get("formula"):
                        generator.population.append(cand["formula"])
                for k in keys: state.r.delete(k)

            # LLM suggestions (OpenAI-only)
            if cfg.get("llm",{}).get("enable", True):
                k = int(cfg["llm"].get("max_suggestions_per_cycle",3))
                for f in proposer.propose(base_features, k=k):
                    if f not in generator.population: generator.population.append(f)

            # Build panel cache
            eth_df = _lh_get_data(lakehouse, cfg, "ETH_USD_SPOT", tf)
            df_cache = {}
            bases = list({canon.split('_')[0] for _, canon in tradeable})
            chosen_bases = random.sample(bases, min(max(1,cfg["discovery"]["panel_symbols"]), len(bases)))
            for base in chosen_bases:
                canon = f"{base}_USD_SPOT"
                df = _lh_get_data(lakehouse, cfg, canon, tf)
                if df.empty: continue
                df_feat = feature_factory.create(df, symbol=canon, cfg=cfg,
                                                 other_dfs={'BTC_USD_SPOT': btc_df, 'ETH_USD_SPOT': eth_df})
                for f in base_features:
                    df_feat[f] = lagged_zscore(df_feat[f], lookback=200)
                df_feat = df_feat.dropna()
                df_feat.attrs["display"] = canon
                df_cache[canon] = df_feat

            # Fitness via panel CV stats
            fitness = {}
            for formula in generator.population:
                fid = hashlib.sha256(json.dumps(formula).encode()).hexdigest()
                panel_returns = {}
                for base in chosen_bases:
                    canon = f"{base}_USD_SPOT"; dff = df_cache.get(canon)
                    if dff is None or dff.empty: continue
                    stats = simlab.run_backtest(dff, formula)
                    overall = stats.get('all', {})
                    if overall and overall.get('n_trades',0)>0:
                        panel_returns[canon] = pd.Series(overall.get('returns', []), dtype=float)
                if not panel_returns:
                    fitness[fid] = 0.0; continue
                cv = panel_cv_stats(panel_returns, k_folds=cfg['discovery']['cv_folds'],
                                    embargo=cfg['discovery']['cv_embargo_bars'],
                                    alpha_fdr=cfg['discovery']['fdr_alpha'])
                fitness[fid] = float(cv['mean_oos'])

            # Promotion pass: BH-FDR + hard gates + factor-aware penalty
            rep_canon = next(iter(df_cache.keys()), None)
            promoted_candidates = []
            if rep_canon:
                df_rep = df_cache.get(rep_canon)
                prom = []
                for formula in generator.population:
                    fid = hashlib.sha256(json.dumps(formula).encode()).hexdigest()
                    if df_rep is None: continue
                    stats_rep = simlab.run_backtest(df_rep, formula)
                    overall = stats_rep.get('all', {})
                    if not overall or overall.get('n_trades',0)==0: continue
                    # pseudo p from fitness
                    pv = 0.5 if fitness.get(fid,0.0)<=0 else max(0.0, 1.0 - min(1.0, fitness[fid]*10))
                    # factor penalty vs current active alphas
                    active = state.get_active_alphas()
                    corr_pen = 0.0
                    if active:
                        cand = np.array(overall.get('returns', []), dtype=float)
                        for a in active:
                            ar = np.array((a.get('performance',{}).get('all',{}).get('returns',[])), dtype=float)
                            if len(ar)>5 and len(cand)>5:
                                m = min(len(ar), len(cand))
                                c = np.corrcoef(ar[-m:], cand[-m:])[0,1]
                                if np.isfinite(c): corr_pen = max(corr_pen, abs(float(c)))
                    if corr_pen >= cfg['discovery']['factor_correlation_max']:
                        # Penalize p-value heavily to avoid crowding
                        pv = min(1.0, pv + 0.4)
                    prom.append({"fid": fid, "formula": formula, "p": pv, "overall": overall})
                if prom:
                    pvals = [c["p"] for c in prom]
                    kept_mask, _ = benjamini_hochberg(pvals, cfg['discovery']['fdr_alpha'])
                    promoted_candidates = [c for c, keep in zip(prom, kept_mask) if keep]

            # Promotion debt guard per UTC day
            day_key = _today_key("promotions")
            already = int(state.get(day_key) or 0)
            budget_left = max(0, int(cfg['discovery']['max_promotions_per_day']) - already)

            # Hard gates + prober + lane tag
            active_promos = []
            gate = cfg['discovery']['promotion_criteria']
            min_rob = float(cfg['prober'].get('min_robust_score', 0.55))
            moon_base_set = set(state.get("lane:moonshot:gainers") or [])
            for c in sorted(promoted_candidates, key=lambda x: x['overall']['sortino'], reverse=True):
                if budget_left <= 0: break
                ov = c['overall']
                if ov.get('n_trades',0) >= gate['min_trades'] and \
                   ov.get('sortino',0) >= gate['min_sortino'] and \
                   ov.get('sharpe',0)  >= gate['min_sharpe']  and \
                   ov.get('win_rate',0)>= gate['min_win_rate'] and \
                   abs(ov.get('mdd',1.0)) <= gate['max_mdd']:
                    rob = prober.score(df_rep, c['formula'])
                    if rob.get("robust_score",0.0) < min_rob: continue
                    base = rep_canon.split('_')[0]
                    chosen_inst = next((inst for inst,_ in tradeable if inst['base'] == base), tradeable[0][0])
                    lane = "moonshot" if base in moon_base_set else "core"
                    alpha = Alpha(
                        id=c['fid'], name=f"alpha_{c['fid'][:6]}", formula=c['formula'],
                        universe=[f"{chosen_inst['base']}_USD_SPOT"], instrument=chosen_inst,
                        timeframe=tf, lane=lane,
                        performance=simlab.run_backtest(df_rep, c['formula']) if df_rep is not None else {}
                    )
                    active_promos.append(alpha.dict()); budget_left -= 1
                    # gene tracking
                    def _track(node):
                        if not isinstance(node[0], list):
                            selfeat, op, _thr = node
                            selfeat = str(selfeat).replace(" ","")
                            g = f"{selfeat}_{op}"
                            state.gene_incr(g, ov.get('sortino',0))
                        else:
                            _track(node[0]); _track(node[2])
                    _track(c['formula'])
                    if len(active_promos) >= cfg['discovery']['max_promotions_per_cycle']:
                        break

            # evolve next generation
            generator.run_evolution_cycle(fitness)

            # Merge promotions
            active = state.get_active_alphas()
            seen = {a['id'] for a in active}
            for a in active_promos:
                if a['id'] not in seen: active.append(a)
            state.set(day_key, (already + len(active_promos)))

            # regime proxy
            regime = 'chop'
            if btc_df is not None and not btc_df.empty:
                rets = btc_df['close'].pct_change()
                if len(rets.dropna())>24:
                    regime = 'high_vol' if rets.rolling(24).std().iloc[-1] > rets.rolling(24).std().quantile(0.75) else 'chop'

            # Optimize → apply lane budgets → risk → execute (paper)
            tw_raw = optimizer.get_weights(active, regime)
            tw = lane_mgr.apply_lane_budgets(tw_raw, active)  # scale per lane budgets
            state.set("target_weights", tw)
            exec_handler.reconcile(tw, active)
            state.set_active_alphas(active)
            state.log_historical_equity(state.get_portfolio().get('equity',0.0))

            logger.info(f"Cycle done. Sleeping {cfg['system']['loop_interval_seconds']}s.")
            time.sleep(cfg["system"]["loop_interval_seconds"])

        except KeyboardInterrupt:
            logger.warning("Shutdown requested."); break
        except Exception as e:
            logger.opt(exception=True).error(f"Loop error: {e}")
            if risk: risk.note_error()
            time.sleep(cfg["system"]["loop_interval_seconds"]*2)

if __name__ == "__main__":
    cli()

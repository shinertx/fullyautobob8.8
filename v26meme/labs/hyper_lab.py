import time, json, random, hashlib
import click, yaml
import pandas as pd
from loguru import logger
from pathlib import Path

from v26meme.core.state import StateManager
from v26meme.data.lakehouse import Lakehouse
from v26meme.research.feature_factory import FeatureFactory
from v26meme.research.generator import GeneticGenerator
from v26meme.labs.simlab import SimLab
from v26meme.research.validation import panel_cv_stats, deflated_sharpe_ratio, benjamini_hochberg
from typing import List, Dict, Tuple, Any

def load_config(file="configs/config.yaml"):
    with open(file, "r") as f: return yaml.safe_load(f)

def _parse_tf_bars(tf: str, days: int) -> int:
    if tf.endswith('h'):
        return int(24//max(1,int(tf[:-1])))*days
    if tf.endswith('m'):
        m = int(tf[:-1]); per_day = 24*60//max(1,m); return per_day*days
    return 24*days

def _extract_features(formula: list) -> List[str]:
    """Extract leaf feature names from nested boolean formula.

    PIT: Pure structural traversal; deterministic.
    """
    feats: List[str] = []
    if not isinstance(formula, list) or not formula:
        return feats
    if not isinstance(formula[0], list):
        # leaf node shape: [feature, op, threshold]
        if len(formula) >= 1 and isinstance(formula[0], str):
            feats.append(formula[0])
        return feats
    feats.extend(_extract_features(formula[0]))
    feats.extend(_extract_features(formula[2]))
    return feats

@click.group()
def cli(): pass

@cli.command()
def run():
    cfg = load_config()
    if not cfg.get('eil',{}).get('enabled', True):
        logger.info("EIL disabled; exiting."); return

    random.seed(cfg['system'].get('seed', 1337))
    state = StateManager(cfg['system']['redis_host'], cfg['system']['redis_port'])
    lake = Lakehouse(preferred_exchange=cfg["execution"]["primary_exchange"])
    ff = FeatureFactory()
    sim = SimLab(cfg['execution']['paper_fees_bps'], cfg['execution']['paper_slippage_bps'],
                 slippage_table=(state.get("slippage:table") or {}))

    base_features = ['return_1p','volatility_20p','momentum_10p','rsi_14','close_vs_sma50',
                     'hod_sin','hod_cos','round_proximity','btc_corr_20p','eth_btc_ratio']
    gen = GeneticGenerator(base_features, population_size=cfg['discovery']['population_size'], seed=cfg['system'].get('seed', 1337))
    tf = "1h"
    nbars = _parse_tf_bars(tf, cfg['eil']['fast_window_days'])

    # Adaptive knobs / gates
    cv_folds = int(cfg['discovery']['cv_folds'])
    cv_embargo = int(cfg['discovery']['cv_embargo_bars'])
    fdr_alpha = float(cfg['discovery']['fdr_alpha'])
    dsr_cfg = cfg.get('validation', {}).get('dsr', {})
    dsr_enabled = bool(dsr_cfg.get('enabled', True))
    dsr_min_prob = float(dsr_cfg.get('min_prob', 0.60))
    dsr_bench = float(dsr_cfg.get('benchmark_sr', 0.0))
    generations_per_cycle = int(cfg['discovery'].get('generations_per_cycle', 1))
    survivor_top_k = int(cfg['eil']['survivor_top_k'])
    promotion_criteria = cfg['discovery'].get('promotion_criteria', {})
    min_trades_gate = int(promotion_criteria.get('min_trades', 50))
    max_promotions_cycle = int(cfg['discovery'].get('max_promotions_per_cycle', 0))
    max_promotions_day = int(cfg['discovery'].get('max_promotions_per_day', 0))

    def _promote_survivors(candidates: Dict[str, Dict[str, Any]], panel_syms: List[str]):
        if not candidates:
            return 0
        active = state.get_active_alphas()
        existing_ids = {a.get('id') for a in active}
        # daily counter
        day_key = time.strftime('%Y%m%d', time.gmtime())
        day_count_key = f"eil:promotions:day:{day_key}"
        day_count_raw = state.get(day_count_key)
        try:
            day_count = int(day_count_raw) if day_count_raw is not None else 0
        except Exception:
            day_count = 0
        promoted = 0
        for key, payload in sorted(candidates.items(), key=lambda kv: kv[1].get('score',0), reverse=True):
            if max_promotions_cycle and promoted >= max_promotions_cycle:
                break
            if max_promotions_day and day_count >= max_promotions_day:
                break
            fid_val = payload.get('fid')
            if not isinstance(fid_val, str):
                continue
            fid = fid_val
            if fid in existing_ids:
                continue
            # Build alpha record (paper mode, moonshot lane initial)
            alpha = {
                'id': fid,
                'name': f"eil_{fid[:8]}",
                'formula': payload.get('formula'),
                'universe': panel_syms,
                'timeframe': tf,
                'lane': 'moonshot',
                'performance': {}
            }
            active.append(alpha)
            existing_ids.add(fid)
            promoted += 1
            day_count += 1
        if promoted:
            state.set_active_alphas(active)
            state.set(day_count_key, day_count)
            logger.info(f"PROMOTED {promoted} survivors -> active_alphas (day_total={day_count})")
        return promoted

    def _drain_llm_proposals(max_inject: int) -> int:
        """Drain hardened LLM proposal queue and append unique formulas to population.

        PIT: Only uses present queue contents; no future leakage. Deterministic insertion order.
        """
        r = getattr(state, 'r', None)
        if r is None:
            return 0
        injected = 0
        seen = set()
        # precompute existing hashes
        for f in gen.population:
            try:
                seen.add(hashlib.sha256(json.dumps(f, separators=(',',':')).encode()).hexdigest())
            except Exception:
                continue
        max_pop = int(cfg['discovery'].get('max_population_size', len(gen.population)))
        while injected < max_inject and r.llen('llm:proposals') > 0:
            raw = r.lpop('llm:proposals')
            if not raw:
                break
            try:
                f = json.loads(raw)
            except Exception:
                continue
            if not isinstance(f, list):
                continue
            # basic shape check
            if len(f) < 3:
                continue
            h = hashlib.sha256(json.dumps(f, separators=(',',':')).encode()).hexdigest()
            if h in seen:
                continue
            gen.population.append(f)
            seen.add(h)
            injected += 1
            if len(gen.population) > max_pop:
                # trim oldest non-injected (keep the most recent injected tail)
                drop_n = len(gen.population) - max_pop
                if drop_n > 0:
                    gen.population = gen.population[drop_n:]
        if injected:
            logger.info(f"LLM_INJECT injected={injected} population={len(gen.population)}")
        return injected

    while True:
        try:
            avail = lake.get_available_symbols(tf)
            if not avail: time.sleep(10); continue
            # Drain LLM proposals early each outer cycle
            _drain_llm_proposals(max_inject=50)
            bases = [s for s in avail if s.endswith("_USD_SPOT")]
            if not bases: time.sleep(10); continue
            panel = random.sample(bases, min(cfg['discovery']['panel_symbols'], len(bases)))

            # Build panel dataframe cache once per outer cycle for determinism
            df_cache: Dict[str, pd.DataFrame] = {}
            btc = lake.get_data("BTC_USD_SPOT", tf)
            eth = lake.get_data("ETH_USD_SPOT", tf)
            for canon in panel:
                df = lake.get_data(canon, tf).tail(nbars)
                if df.empty: continue
                df_feat = ff.create(df, symbol=canon, cfg=cfg, other_dfs={'BTC_USD_SPOT': btc, 'ETH_USD_SPOT': eth})
                df_cache[canon] = df_feat.dropna()

            if not gen.population: gen.initialize_population()

            cycle_promoted = 0
            for _gen in range(generations_per_cycle):
                fitness_scores: Dict[str, float] = {}
                formula_stats: Dict[str, Dict[str, float]] = {}
                pvals: List[float] = []
                fids: List[str] = []
                raw_records: List[Tuple[str, list, Dict[str, list]]] = []  # fid, formula, panel_returns dict

                for f in gen.population:
                    fid = hashlib.sha256(json.dumps(f).encode()).hexdigest()
                    panel_returns: Dict[str, pd.Series] = {}
                    trade_counts: List[int] = []
                    sharpe_vals: List[float] = []
                    all_returns_concat: List[float] = []
                    for canon, dff in df_cache.items():
                        stats = sim.run_backtest(dff, f).get("all", {})
                        if stats and stats.get("n_trades",0) > 0:
                            trade_counts.append(int(stats.get("n_trades",0)))
                            sharpe_vals.append(float(stats.get("sharpe",0.0)))
                            r_list = stats.get("returns", [])
                            if isinstance(r_list, list) and r_list:
                                ser = pd.Series(r_list, dtype=float)
                                panel_returns[canon] = ser
                                all_returns_concat.extend(ser.tolist())
                    if not panel_returns:
                        continue
                    # Cross-validation p-value (panel aggregated)
                    cv_res = panel_cv_stats(panel_returns, k_folds=cv_folds, embargo=cv_embargo, alpha_fdr=fdr_alpha, cv_method=cfg['discovery'].get('cv_method','kfold'))
                    p_val = float(cv_res.get('p_value', 1.0))
                    mean_oos = float(cv_res.get('mean_oos', 0.0))
                    # Deflated Sharpe probability
                    dsr_prob = 0.0
                    if all_returns_concat and dsr_enabled:
                        dsr_prob = deflated_sharpe_ratio(pd.Series(all_returns_concat, dtype=float), n_trials=len(gen.population), sr_benchmark=dsr_bench)
                    elif all_returns_concat:
                        dsr_prob = 1.0  # pass-through if disabled
                    total_trades = sum(trade_counts)
                    mean_sharpe = sum(sharpe_vals)/max(1,len(sharpe_vals))
                    # Risk-adjusted fitness: prefer higher Sharpe & breadth; penalize insufficient trades.
                    trade_factor = min(1.0, total_trades / max(1, min_trades_gate))
                    fitness = mean_sharpe * trade_factor
                    fitness_scores[fid] = fitness
                    formula_stats[fid] = {"p_value": p_val, "mean_oos": mean_oos, "dsr_prob": dsr_prob, "trades": float(total_trades), "fitness": fitness}
                    pvals.append(p_val)
                    fids.append(fid)
                    raw_records.append((fid, f, {k: v.tolist() for k,v in panel_returns.items()}))

                # FDR gate across current generation
                accepts = []
                if pvals:
                    accept_flags, fdr_thresh = benjamini_hochberg(pvals, fdr_alpha)
                    accepts = [f for f, flag in zip(fids, accept_flags) if flag]
                else:
                    accepts = []

                accepted_payload: Dict[str, Dict[str, Any]] = {}
                survivors_logged = 0
                # Collect survivors meeting all gates
                gated = []
                for fid, formula, panel_ret_dict in raw_records:
                    stats = formula_stats.get(fid, {})
                    if not stats: continue
                    if fid not in accepts:
                        continue
                    if dsr_enabled and stats.get('dsr_prob',0.0) < dsr_min_prob:
                        continue
                    if stats.get('trades',0.0) < min_trades_gate:
                        continue
                    gated.append((stats.get('fitness',0.0), fid, formula, stats))
                gated.sort(key=lambda x: x[0], reverse=True)
                topk = gated[:survivor_top_k]

                # Persist survivors
                now_ts = int(time.time())
                for fitness, fid, form, stats in topk:
                    payload = {"fid": fid, "formula": form, "score": fitness, "p_value": stats.get('p_value'),
                               "dsr_prob": stats.get('dsr_prob'), "trades": stats.get('trades'), "ts": now_ts}
                    accepted_payload[f"eil:candidates:{fid}"] = payload
                    # gene usage tracking
                    for feat in _extract_features(form):
                        state.gene_incr(feat, fitness)
                    logger.info(f"SURVIVOR fid={fid} fitness={fitness:.4f} p={stats.get('p_value'):.4g} dsr={stats.get('dsr_prob'):.3f} trades={stats.get('trades')}")
                    survivors_logged += 1
                if accepted_payload:
                    state.multi_set(accepted_payload)
                    # Promotion bridge (conveyor) once per generation (respect cycle & day caps)
                    cycle_promoted += _promote_survivors(accepted_payload, panel)
                else:
                    logger.info("SURVIVOR_NONE generation - no candidates passed gates")

                # Evolve population for next generation deterministically
                gen.run_evolution_cycle(fitness_scores)
                # Short deterministic pause between generations to throttle compute
                if generations_per_cycle > 1 and _gen < generations_per_cycle - 1:
                    time.sleep(1)

            if cycle_promoted:
                logger.info(f"PROMOTION_CYCLE total_promoted={cycle_promoted}")
            # Cycle sleep
            time.sleep(15)
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.opt(exception=True).error(f"EIL loop error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    cli()

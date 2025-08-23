import time, json, random, hashlib
import os
import click, yaml
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
import math  # FIX: for activation_gain exp (replaces deprecated pd.np)

from v26meme.core.state import StateManager
from v26meme.data.lakehouse import Lakehouse
from v26meme.research.feature_factory import FeatureFactory
from v26meme.research.generator import GeneticGenerator  # updated: adaptive generator w/ feature stats
from v26meme.labs.simlab import SimLab
from v26meme.research.validation import panel_cv_stats, deflated_sharpe_ratio, benjamini_hochberg
from typing import List, Dict, Tuple, Any

# NEW: helper to apply single-symbol restriction
def _maybe_restrict_single_symbol(cfg: dict, symbols: List[str]) -> List[str]:
    harv = cfg.get('harvester', {})
    core = harv.get('core_symbols') or []
    restrict = bool(harv.get('restrict_single_symbol', False))
    if restrict and isinstance(core, list) and len(core) == 1:
        target = core[0]
        return [s for s in symbols if s == target]
    return symbols

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

def _select_timeframe(cfg: dict, lake: Lakehouse, min_bars: int, min_panel: int, prefer: List[str]) -> Tuple[str | None, List[str], Dict[str, int]]:
    """Deterministically select the highest timeframe meeting coverage gates.

    PIT: Uses only already-harvested bar counts (closed history). No forward data access.
    Returns (timeframe | None if none qualifies, eligible_symbols, bars_map).
    """
    chosen_tf: str | None = None
    eligible: List[str] = []
    bars_map: Dict[str, int] = {}
    for tf in prefer:
        avail = lake.get_available_symbols(tf)
        if not avail:
            continue
        # NEW: optional single-symbol narrowing (consistent with main loop gating)
        avail = _maybe_restrict_single_symbol(cfg, avail)
        cur_bars = {s: len(lake.get_data(s, tf)) for s in avail}
        cur_eligible = [s for s, n in cur_bars.items() if n >= min_bars]
        if len(cur_eligible) >= min_panel:
            chosen_tf = tf
            eligible = sorted(cur_eligible)
            bars_map = cur_bars
            break
    return chosen_tf, eligible, bars_map

def _sleep(seconds: int) -> None:
    """Deterministic sleep wrapper honoring test override.

    If EIL_SLEEP_OVERRIDE is set (any value), sleep is skipped to keep tests fast.
    PIT: does not alter logic—only temporal pacing side-effect.
    """
    if os.environ.get('EIL_SLEEP_OVERRIDE') is not None:
        return
    time.sleep(seconds)

@click.group()
def cli(): pass

@cli.command()
def run():  # Plain function for programmatic/test use
    run_eil()

def run_eil(cfg: Dict[str, Any] | None = None) -> None:
    """Execute one or more Evolutionary Iteration Loop (EIL) cycles.

    This is the programmatic (non-Click) entrypoint used by tests to avoid Click
    argument parsing side-effects (pytest -q etc.). The Click CLI command `run`
    delegates to this function.

    Args:
        cfg: Optional pre-loaded configuration dict. If None, load_config() used.

    PIT NOTE: Operates solely on already-harvested lakehouse data; no forward leakage.
    """
    if cfg is None:
        cfg = load_config()
    # Ensure base keys exist (minimal type guard for mypy/pylance)
    assert isinstance(cfg, dict) and 'system' in cfg and 'discovery' in cfg and 'execution' in cfg and 'eil' in cfg
    # Initialize rejection telemetry key if absent (tests rely on presence even if no generation)
    state = StateManager(cfg['system']['redis_host'], cfg['system']['redis_port'])
    if state.get('eil:rej:counts') is None:
        state.set('eil:rej:counts', {})
    # Relative cycle baseline (fix: treat EIL_MAX_CYCLES as delta not absolute)
    start_idx = int(state.get('eil:cycle_idx') or 0)
    if os.environ.get('EIL_FRESH') == '1' and state.get('eil:population_struct') is None:
        # Fresh sandbox run without destroying long-term telemetry if population exists
        state.set('eil:cycle_idx', 0)
        start_idx = 0
        logger.info("EIL_FRESH_RESET applied (population_struct absent)")
    # reassign after early state creation
    lake = Lakehouse(preferred_exchange=cfg["execution"]["primary_exchange"])
    ff = FeatureFactory()
    sim = SimLab(cfg['execution']['paper_fees_bps'], cfg['execution']['paper_slippage_bps'],
                 slippage_table=(state.get("slippage:table") or {}),
                 max_holding_bars=int(cfg['execution'].get('max_holding_bars', 0) or 0) or None)

    base_features = ['return_1p','volatility_20p','momentum_10p','rsi_14','close_vs_sma50',
                     'hod_sin','hod_cos','round_proximity','btc_corr_20p','eth_btc_ratio','beta_btc_20p']
    # Coverage gating parameters (avoid magic numbers by sourcing config)
    min_panel_symbols = int(cfg['discovery'].get('min_panel_symbols', 3))
    min_bars_per_symbol = int(cfg['discovery'].get('min_bars_per_symbol', 100))
    timeframe_preference = ['1h','15m','5m']  # higher frequency fallback later if higher tf unseeded

    tf_selected, eligible_symbols, bars_map = _select_timeframe(cfg, lake, min_bars_per_symbol, min_panel_symbols, timeframe_preference)
    # Log if narrowing applied
    harv_cfg = cfg.get('harvester', {})
    if harv_cfg.get('restrict_single_symbol'):
        logger.info(f"EIL_SINGLE_SYMBOL_MODE core_symbols={harv_cfg.get('core_symbols')} eligible_after_narrow={len(eligible_symbols)}")

    gen = GeneticGenerator(base_features, population_size=cfg['discovery']['population_size'], seed=cfg['system'].get('seed', 1337))
    # Warm start population if persisted
    try:
        persisted = state.get('eil:population_struct')
        if isinstance(persisted, list) and persisted:
            # Validate structure (list of lists)
            valid = [f for f in persisted if isinstance(f, list)]
            if valid:
                gen.population = valid[:gen.population_size]
                logger.info(f"EIL_POPULATION_WARM reloaded={len(gen.population)}")
    except Exception:
        pass

    # ---------------- inner helpers (promotion unchanged) ----------------
    def _promote_survivors(candidates: Dict[str, Dict[str, Any]], panel_syms: List[str]):
        if not candidates:
            return 0
        active = state.get_active_alphas()
        existing_ids = {a.get('id') for a in active}
        day_key = time.strftime('%Y%m%d', time.gmtime())
        day_count_key = f"eil:promotions:day:{day_key}"
        day_count_raw = state.get(day_count_key)
        try: day_count = int(day_count_raw) if day_count_raw is not None else 0
        except Exception: day_count = 0
        promoted = 0
        for key, payload in sorted(candidates.items(), key=lambda kv: kv[1].get('score',0), reverse=True):
            if max_promotions_cycle and promoted >= max_promotions_cycle: break
            if max_promotions_day and day_count >= max_promotions_day: break
            fid_val = payload.get('fid')
            if not isinstance(fid_val, str): continue
            fid = fid_val
            if fid in existing_ids: continue
            alpha = {
                'id': fid,
                'name': f"eil_{fid[:8]}",
                'formula': payload.get('formula'),
                'universe': panel_syms,
                'timeframe': tf_selected,
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
        if r is None: return 0
        injected = 0
        seen = set()
        for f in gen.population:
            try: seen.add(hashlib.sha256(json.dumps(f, separators=(',',':')).encode()).hexdigest())
            except Exception: continue
        max_pop = int(cfg['discovery'].get('max_population_size', len(gen.population)))
        while injected < max_inject and r.llen('llm:proposals') > 0:
            raw = r.lpop('llm:proposals')
            if not raw: break
            try: f = json.loads(raw)
            except Exception: continue
            if not isinstance(f, list) or len(f) < 3: continue
            h = hashlib.sha256(json.dumps(f, separators=(',',':')).encode()).hexdigest()
            if h in seen: continue
            gen.population.append(f)
            seen.add(h)
            injected += 1
            if len(gen.population) > max_pop:
                drop_n = len(gen.population) - max_pop
                if drop_n > 0:
                    gen.population = gen.population[drop_n:]
        if injected:
            logger.info(f"LLM_INJECT injected={injected} population={len(gen.population)}")
        return injected

    # ---------------- main loop ----------------
    while True:
        max_cycles_env = int(os.environ.get('EIL_MAX_CYCLES','0') or 0)
        cycle_idx = int(state.get('eil:cycle_idx') or 0)
        rel_cycles = cycle_idx - start_idx
        if max_cycles_env and rel_cycles >= max_cycles_env:
            logger.info(f"EIL_RELATIVE_COMPLETE rel_cycles={rel_cycles} max={max_cycles_env} start_idx={start_idx} final_cycle_idx={cycle_idx}")
            break
        try:
            logger.info(f"EIL_CYCLE_START idx={cycle_idx} rel={rel_cycles}")
            # Re-evaluate timeframe & coverage each outer cycle for adaptive promotion to richer TF.
            tf_selected, eligible_symbols, bars_map = _select_timeframe(cfg, lake, min_bars_per_symbol, min_panel_symbols, timeframe_preference)
            if tf_selected is None:
                state.set('eil:diag:last_reason', f"coverage_gate unmet min_symbols={min_panel_symbols} min_bars={min_bars_per_symbol}")
                _sleep(1)
                state.set('eil:cycle_idx', cycle_idx + 1)
                continue
            # Drain LLM queue early
            _drain_llm_proposals(max_inject=50)

            # Panel sampling from eligible symbol pool
            if not eligible_symbols:
                logger.info("EIL_PANEL_EMPTY eligible_symbols=0")
                _sleep(1); state.set('eil:cycle_idx', cycle_idx + 1); continue
            panel_universe = [s for s in eligible_symbols if s.endswith('_USD_SPOT')]
            if not panel_universe:
                logger.info("EIL_PANEL_EMPTY after _USD_SPOT filter")
                _sleep(1); state.set('eil:cycle_idx', cycle_idx + 1); continue
            panel = random.sample(panel_universe, min(cfg['discovery']['panel_symbols'], len(panel_universe)))

            # Build panel dataframe cache & empirical feature stats (quantiles) once per cycle
            nbars_window = int(_parse_tf_bars(tf_selected, cfg['eil']['fast_window_days']))
            df_cache: Dict[str, pd.DataFrame] = {}
            feat_samples: Dict[str, list] = {f: [] for f in base_features}
            btc_all = lake.get_data('BTC_USD_SPOT', tf_selected)
            eth_all = lake.get_data('ETH_USD_SPOT', tf_selected)
            for canon in panel:
                raw_df = lake.get_data(canon, tf_selected).tail(nbars_window)
                if raw_df.empty: continue
                dff = ff.create(raw_df, symbol=canon, cfg=cfg, other_dfs={'BTC_USD_SPOT': btc_all, 'ETH_USD_SPOT': eth_all})
                # PREVIOUSLY: unconditional dff = dff.dropna() caused full wipe because early-window feature NaNs.
                # New: only require core OHLCV columns to be present; allow feature NaNs (handled by feature gating).
                core_cols = [c for c in ['open','high','low','close','volume'] if c in dff.columns]
                before_rows = len(dff)
                if core_cols:
                    dff = dff.dropna(subset=core_cols)
                after_rows = len(dff)
                # Optional row-level feature completeness threshold (configurable) – default 0 (disabled)
                row_min_ratio = float(cfg['discovery'].get('row_min_feature_non_nan_ratio', 0.0))
                if row_min_ratio > 0:
                    feat_cols = [c for c in base_features if c in dff.columns]
                    if feat_cols:
                        nn_counts = dff[feat_cols].notna().sum(axis=1)
                        needed = max(1, int(len(feat_cols) * row_min_ratio))
                        dff = dff.loc[nn_counts >= needed]
                if dff.empty:
                    logger.debug(f"EIL_PANEL_DROP canon={canon} emptied rows_before={before_rows} after_core_drop={after_rows}")
                    continue
                if after_rows and after_rows < before_rows:
                    logger.debug(f"EIL_PANEL_TRIM canon={canon} kept={after_rows}/{before_rows} core_cols={core_cols}")
                df_cache[canon] = dff
                # Collect feature samples (head/tail slice to bound memory)
                for f in base_features:
                    if f in dff:
                        col = dff[f].dropna()
                        if not col.empty:
                            step = max(1, len(col)//50)
                            feat_samples[f].extend(col.iloc[::step].tolist())
            if not df_cache:
                logger.info("EIL_DF_CACHE_EMPTY panel=%s tf=%s" % (panel, tf_selected))
                state.set('eil:diag:last_reason', 'no_df_cache_after_feature_build')
                _sleep(1)
                state.set('eil:cycle_idx', cycle_idx + 1)
                continue
            else:
                logger.info(f"EIL_DF_CACHE_OK tf={tf_selected} panel={panel} lens={[len(df_cache[s]) for s in df_cache]}")
            # Compute empirical stats & inject into generator
            feature_stats: Dict[str, Dict[str, float]] = {}
            # Feature continuity / variance gating (adaptive, config-driven)
            min_non_nan = float(cfg['discovery'].get('feature_min_non_nan_ratio', 0.30))
            min_var = float(cfg['discovery'].get('feature_min_variance', 1e-6))
            active_feature_mask: Dict[str, bool] = {}
            for f, vals in feat_samples.items():
                if not vals:
                    continue
                s = pd.Series(vals, dtype=float)
                nn_ratio = float(s.notna().mean())
                var_raw = s.var(ddof=1) if len(s) > 1 else 0.0
                try:
                    var_val = float(var_raw) if isinstance(var_raw, (int,float)) and not pd.isna(var_raw) else 0.0
                except Exception:
                    var_val = 0.0
                keep = (nn_ratio >= min_non_nan) and (var_val >= min_var)
                active_feature_mask[f] = keep
                if keep:
                    q_low = float(s.quantile(0.10))
                    q_high = float(s.quantile(0.90))
                    feature_stats[f] = {
                        'q_low': q_low,
                        'q_high': q_high,
                        'min': float(s.min()),
                        'max': float(s.max())
                    }
            # Log diagnostic summary once per cycle
            diag = {f: {'keep': active_feature_mask.get(f, False)} for f in base_features}
            state.set('eil:feature_gate_diag', diag)
            state.set('eil:feature_stats', {f: {**feature_stats.get(f, {}), 'keep': active_feature_mask.get(f, False)} for f in base_features})
            # Prune generator feature list only for new condition sampling; existing formulas preserved
            pruned_features = [f for f in gen.features if active_feature_mask.get(f, False)] or gen.features
            gen.features = pruned_features
            # Population hygiene: reseed fraction of formulas containing suppressed features
            suppressed = {f for f,k in active_feature_mask.items() if not k}
            # Feature continuity counters
            cont_key_active = 'eil:feat:active_ct'
            cont_key_cycles = 'eil:feat:cycles_ct'
            cont_active = state.get(cont_key_active) or {}
            cont_cycles = state.get(cont_key_cycles) or {}
            for f in base_features:
                cont_cycles[f] = int(cont_cycles.get(f,0)) + 1
                if active_feature_mask.get(f, False):
                    cont_active[f] = int(cont_active.get(f,0)) + 1
            state.set(cont_key_active, cont_active)
            state.set(cont_key_cycles, cont_cycles)
            continuity_threshold = float(cfg['discovery'].get('adaptive', {}).get('continuity_threshold', 0.80))
            patience = int(cfg['discovery'].get('adaptive', {}).get('continuity_suppression_patience', 5))
            continuity_suppress = set()
            continuity_ratios = {}
            for f in base_features:
                a = cont_active.get(f,0); c = cont_cycles.get(f,0)
                ratio = a / c if c>0 else 0.0
                continuity_ratios[f] = ratio
                if c >= patience and ratio < continuity_threshold:
                    continuity_suppress.add(f)
            if continuity_suppress:
                state.set('eil:continuity_suppress', list(continuity_suppress))
            suppressed |= continuity_suppress
            if suppressed:
                affected_idx = []
                for i, form in enumerate(gen.population):
                    used = set(_extract_features(form))
                    if used & suppressed:
                        affected_idx.append(i)
                if affected_idx:
                    reseed_fraction = float(cfg['discovery'].get('reseed_fraction', 0.30))
                    n_reseed = max(1, int(len(affected_idx) * reseed_fraction))
                    for idx in affected_idx[:n_reseed]:
                        gen.population[idx] = gen._create_random_formula()
                    state.set('eil:population_hygiene', {'suppressed': list(suppressed), 'continuity_suppress': list(continuity_suppress), 'affected': len(affected_idx), 'reseeded': n_reseed})
            state.set('eil:feature_continuity', continuity_ratios)

            if not gen.population:
                gen.initialize_population()
                logger.info(f"EIL_POP_INIT size={len(gen.population)}")

            # Adaptive gate params (existing)
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
            global min_sortino, min_sharpe, min_win_rate, max_mdd, max_promotions_cycle, max_promotions_day
            min_sortino = float(promotion_criteria.get('min_sortino', 0.0))
            min_sharpe = float(promotion_criteria.get('min_sharpe', 0.0))
            min_win_rate = float(promotion_criteria.get('min_win_rate', 0.0))
            max_mdd = float(promotion_criteria.get('max_mdd', 1.0))
            max_promotions_cycle = int(cfg['discovery'].get('max_promotions_per_cycle', 0))
            max_promotions_day = int(cfg['discovery'].get('max_promotions_per_day', 0))

            cycle_promoted = 0
            for _gen in range(generations_per_cycle):
                logger.info(f"EIL_GEN_START cycle={cycle_idx} gen={_gen} pop={len(gen.population)}")
                fitness_scores: Dict[str, float] = {}
                formula_stats: Dict[str, Dict[str, float]] = {}
                pvals: List[float] = []
                fids: List[str] = []
                raw_records: List[Tuple[str, list, Dict[str, list]]] = []
                zero_trade_formulas = 0

                for f in gen.population:
                    fid = hashlib.sha256(json.dumps(f, separators=(',',':')).encode()).hexdigest()
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
                        zero_trade_formulas += 1
                        continue
                    cv_res = panel_cv_stats(panel_returns, k_folds=cv_folds, embargo=cv_embargo, alpha_fdr=fdr_alpha, cv_method=cfg['discovery'].get('cv_method','kfold'))
                    p_val = float(cv_res.get('p_value', 1.0))
                    mean_oos = float(cv_res.get('mean_oos', 0.0))
                    dsr_prob = 0.0
                    agg_win_rate = 0.0
                    agg_sortino = 0.0
                    agg_sharpe = 0.0
                    agg_mdd = 0.0
                    if all_returns_concat:
                        ser_all = pd.Series(all_returns_concat, dtype=float)
                        # Aggregate performance stats (PIT-safe: only historical trade PnLs)
                        win_rate = float((ser_all > 0).mean())
                        down_std = ser_all[ser_all < 0].std(ddof=1)
                        sortino = float(ser_all.mean() / down_std) if down_std and down_std > 0 else 0.0
                        std_all = ser_all.std(ddof=1)
                        sr = float(ser_all.mean() / std_all) if std_all and std_all > 0 else 0.0
                        eq = (1+ser_all).cumprod()
                        mdd = float(((eq - eq.cummax()) / eq.cummax()).min()) if not eq.empty else 0.0
                        agg_win_rate, agg_sortino, agg_sharpe, agg_mdd = win_rate, sortino, sr, mdd
                        if dsr_enabled:
                            dsr_prob = deflated_sharpe_ratio(ser_all, n_trials=len(gen.population), sr_benchmark=dsr_bench)
                        else:
                            dsr_prob = 1.0
                    total_trades = sum(trade_counts)
                    mean_sharpe = sum(sharpe_vals)/max(1,len(sharpe_vals))
                    trade_factor = min(1.0, total_trades / max(1, min_trades_gate))
                    # Activation gain (encourages evaluable formulas) — exponential saturation (config act_scale default 20)
                    act_scale = float(cfg['discovery'].get('activation_trade_scale', 20.0))
                    activation_gain = 1.0 - math.exp(- total_trades / max(1.0, act_scale)) if total_trades > 0 else 0.0
                    activation_gain = max(0.0, min(1.0, activation_gain))
                    # Risk-aware drawdown penalty (adaptive, config-driven)
                    penalty_scale = float(cfg['discovery'].get('fitness_drawdown_penalty_scale', 0.40))
                    dd_pen = 1.0
                    if agg_mdd < 0:  # mdd negative
                        dd_pen = max(0.0, 1.0 + agg_mdd * penalty_scale)
                    # Feature concentration penalty: average gene weight frequency
                    freq = 0.0
                    try:
                        gene_usage = state.get('gene:usage') or {}
                        used_feats = _extract_features(f)
                        if used_feats:
                            freq = sum(gene_usage.get(feat,0) for feat in used_feats)/max(1,len(used_feats))
                    except Exception:
                        pass
                    conc_pen_scale = float(cfg['discovery'].get('fitness_concentration_penalty_scale', 0.25))
                    conc_pen = 1.0 / (1.0 + conc_pen_scale * freq) if freq>0 else 1.0
                    # Return variance penalty (stability preference)
                    ret_var = 0.0
                    if all_returns_concat:
                        ser_all = pd.Series(all_returns_concat, dtype=float)
                        ret_var_raw = ser_all.var(ddof=1) if ser_all.shape[0] > 1 else 0.0
                        try:
                            ret_var = float(ret_var_raw) if isinstance(ret_var_raw,(int,float)) and not pd.isna(ret_var_raw) else 0.0
                        except Exception:
                            ret_var = 0.0
                    else:
                        ret_var = 0.0
                    var_pen_scale = float(cfg['discovery'].get('fitness_variance_penalty_scale', 0.15))
                    var_pen = 1.0 / (1.0 + var_pen_scale * ret_var) if ret_var>0 else 1.0
                    # Raw profitability signal (gated to be non-negative)
                    profit_signal = max(0.0, agg_sortino)

                    # Trade diversity signal (rewards more trades, with diminishing returns)
                    trade_signal = np.log1p(total_trades)

                    # Combine signals with weights
                    fitness_weights = cfg['discovery'].get('fitness_weights', {'profit_signal': 0.7, 'trade_signal': 0.3})
                    w_profit = fitness_weights.get('profit_signal', 0.7)
                    w_trade = fitness_weights.get('trade_signal', 0.3)

                                        # Raw profitability signal (gated to be non-negative)
                    profit_signal = max(0.0, agg_sortino)

                    # Trade diversity signal (rewards more trades, with diminishing returns)
                    trade_signal = np.log1p(total_trades)

                    # Combine signals with weights
                    fitness_weights = cfg['discovery'].get('fitness_weights', {'profit_signal': 0.7, 'trade_signal': 0.3})
                    w_profit = fitness_weights.get('profit_signal', 0.7)
                    w_trade = fitness_weights.get('trade_signal', 0.3)

                    fitness = (w_profit * profit_signal) + (w_trade * trade_signal)
                    fitness *= dd_pen * conc_pen * var_pen # Apply existing penalties
                    fitness_scores[fid] = fitness
                    formula_stats[fid] = {
                        "p_value": p_val, "mean_oos": mean_oos, "dsr_prob": dsr_prob,
                        "trades": float(total_trades), "fitness": fitness,
                        "win_rate": agg_win_rate, "sortino": agg_sortino, "sharpe": agg_sharpe, "mdd": agg_mdd
                    }
                    pvals.append(p_val)
                    fids.append(fid)
                    raw_records.append((fid, f, {k: v.tolist() for k,v in panel_returns.items()}))
                # After building raw_records, initialize rejection telemetry default
                state.set('eil:rej:counts', state.get('eil:rej:counts') or {})
                logger.info(f"EIL_GEN_POST_EVAL pop={len(gen.population)} zero_trade_formulas={zero_trade_formulas}")
                # Instrument distributions for diagnostics (single-symbol instability insight)
                if formula_stats:
                    try:
                        pv_list = [formula_stats[fid]['p_value'] for fid in formula_stats if 'p_value' in formula_stats[fid]]
                        sh_list = [formula_stats[fid]['sharpe'] for fid in formula_stats if 'sharpe' in formula_stats[fid]]
                        tr_list = [formula_stats[fid]['trades'] for fid in formula_stats if 'trades' in formula_stats[fid]]
                        def _summ(a):
                            if not a: return {}
                            s = pd.Series(a, dtype=float)
                            return {
                                'count': int(s.shape[0]),
                                'min': float(s.min()),
                                'p25': float(s.quantile(0.25)),
                                'median': float(s.median()),
                                'p75': float(s.quantile(0.75)),
                                'max': float(s.max())
                            }
                        pval_stats = _summ(pv_list)
                        sharpe_stats = _summ(sh_list)
                        trade_stats = _summ(tr_list)
                        state.set('eil:diag:last_pval_stats', pval_stats)
                        state.set('eil:diag:last_perf_stats', {'sharpe': sharpe_stats, 'trades': trade_stats})
                        logger.info(f"EIL_DIST pvals_med={pval_stats.get('median')} sharpe_med={sharpe_stats.get('median')} trades_med={trade_stats.get('median')}")
                    except Exception:
                        pass

                # Degeneracy detection (adaptive threshold; derived to avoid magic number)
                pop_evaluable = len(gen.population) - zero_trade_formulas
                deg_pct = zero_trade_formulas / max(1,len(gen.population))
                # Threshold derived: cap at 0.85; scales modestly with population breadth
                degen_threshold = min(0.85, 0.50 + 0.02 * (gen.population_size / max(1,len(base_features))))
                state.set('eil:degenerate_pct', round(deg_pct,4))
                if deg_pct >= degen_threshold:
                    logger.warning(f"EIL_DEGENERATE pct={deg_pct:.2f} >= {degen_threshold:.2f} reseeding population")
                    gen.initialize_population()
                    continue  # restart generation loop with fresh population

                # FDR gate (with optional debug bypass for single-symbol sandbox)
                accepts = []
                fdr_thresh = None
                debug_relaxed = bool(cfg['discovery'].get('debug_relaxed_gates'))
                if pvals:
                    accept_flags, fdr_thresh = benjamini_hochberg(pvals, fdr_alpha)
                    accepts = [f for f, flag in zip(fids, accept_flags) if flag]
                if debug_relaxed and not accepts and pvals:
                    # Diagnostic bypass: accept all to inspect downstream attrition; still record theoretical threshold
                    accepts = fids[:]
                    logger.warning(f"EIL_FDR_DEBUG_BYPASS active total={len(accepts)} fdr_thresh={fdr_thresh}")
                    state.set('eil:diag:last_fdr_bypass', {'active': True, 'fdr_thresh': fdr_thresh, 'accepted': len(accepts)})
                else:
                    state.set('eil:diag:last_fdr_bypass', {'active': False, 'accepted': len(accepts), 'fdr_thresh': fdr_thresh})
                accepted_payload: Dict[str, Dict[str, Any]] = {}
                # Rejection telemetry counters
                rej_counts: Dict[str,int] = {k:0 for k in ['fdr_reject','dsr_fail','trades_fail','sortino_fail','sharpe_fail','winrate_fail','mdd_fail']}
                for fid, formula, panel_ret_dict in raw_records:
                    stats = formula_stats.get(fid, {})
                    if not stats: continue
                    if fid not in accepts:
                        rej_counts['fdr_reject'] += 1
                        continue
                    if dsr_enabled and stats.get('dsr_prob',0.0) < dsr_min_prob:
                        rej_counts['dsr_fail'] += 1; continue
                    if stats.get('trades',0.0) < min_trades_gate:
                        rej_counts['trades_fail'] += 1; continue
                    if stats.get('sortino',0.0) < min_sortino:
                        rej_counts['sortino_fail'] += 1; continue
                    if stats.get('sharpe',0.0) < min_sharpe:
                        rej_counts['sharpe_fail'] += 1; continue
                    if stats.get('win_rate',0.0) < min_win_rate:
                        rej_counts['winrate_fail'] += 1; continue
                    if stats.get('mdd',0.0) < -max_mdd:
                        rej_counts['mdd_fail'] += 1; continue
                    accepted_payload[fid] = {**stats, 'fid': fid, 'formula': formula, 'score': stats.get('fitness',0.0)}
                state.set('eil:rej:counts', rej_counts)
                survivors = sorted(accepted_payload.values(), key=lambda d: d.get('score',0.0), reverse=True)[:survivor_top_k]
                save_map = {}
                now_ts = int(time.time())
                for s_payload in survivors:
                    fid = s_payload['fid']
                    out_payload = {k: s_payload.get(k) for k in ['fid','formula','score','p_value','dsr_prob','trades']}
                    out_payload['ts'] = now_ts
                    save_map[f"eil:candidates:{fid}"] = out_payload
                    for feat in _extract_features(s_payload['formula']):
                        state.gene_incr(feat, s_payload.get('score',0.0))
                    logger.info(f"SURVIVOR fid={fid} fitness={s_payload.get('score',0.0):.4f} p={s_payload.get('p_value'):.3g} dsr={s_payload.get('dsr_prob',0.0):.3f} trades={s_payload.get('trades')}")
                if survivors:
                    # Persist last survivor for external diagnostics
                    try:
                        top = survivors[0]
                        state.set('eil:diag:last_survivor', {k: top.get(k) for k in ['fid','score','p_value','dsr_prob','trades']})
                    except Exception:
                        pass
                if save_map:
                    state.multi_set(save_map)
                    cycle_promoted += _promote_survivors(save_map, panel)
                else:
                    logger.info("SURVIVOR_NONE generation - no candidates passed gates")

                # Evolve population deterministically
                gen.run_evolution_cycle(fitness_scores)
                # Persist population & feature stats for warm restarts
                try:
                    state.set('eil:population_struct', gen.population)
                except Exception:
                    pass
                if generations_per_cycle > 1 and _gen < generations_per_cycle - 1:
                    _sleep(1)

            if cycle_promoted:
                logger.info(f"PROMOTION_CYCLE total_promoted={cycle_promoted}")
            _sleep(1)
            state.set('eil:cycle_idx', cycle_idx + 1)
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.opt(exception=True).error(f"EIL loop error: {e}")
            _sleep(1)

@cli.command('run')
def run_cmd():  # Click CLI entrypoint delegates
    run_eil()

if __name__ == "__main__":
    cli()
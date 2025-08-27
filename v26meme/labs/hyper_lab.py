import time, json, random, hashlib
import os
import yaml
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
import math  # FIX: for activation_gain exp (replaces deprecated pd.np)
import click

from v26meme.core.state import StateManager
from v26meme.data.lakehouse import Lakehouse
from v26meme.research.feature_factory import FeatureFactory
from v26meme.research.generator import GeneticGenerator  # updated: adaptive generator w/ feature stats
from v26meme.labs.simlab import SimLab
from v26meme.research.validation import Validator
from v26meme.research.feature_prober import FeatureProber
from typing import List, Dict, Tuple, Any, Optional

# Stage helper functions (ensure defined before run_eil)

def _apply_stage_overrides(state: StateManager, cfg: dict, current_stage: str, gate_stages_cfg: list) -> tuple[float, float, int]:
    fdr_alpha = float(cfg['discovery']['fdr_alpha'])
    dsr_cfg = cfg.get('validation', {}).get('dsr', {})
    dsr_min_prob = float(dsr_cfg.get('min_prob', 0.60))
    min_trades = int(cfg['discovery'].get('promotion_criteria', {}).get('min_trades', 50))
    stage_map = {s['name']: s for s in gate_stages_cfg if 'name' in s}
    if current_stage in stage_map:
        st = stage_map[current_stage]
        fdr_alpha = float(st.get('fdr_alpha', fdr_alpha))
        dsr_min_prob = float(st.get('dsr_min_prob', dsr_min_prob))
        min_trades = max(min_trades, int(st.get('min_trades', 0)))
    return fdr_alpha, dsr_min_prob, min_trades

def _maybe_escalate_stage(state: StateManager, cycle_idx: int, current_stage: str, gate_stages_cfg: list, stage_escalation_cfg: dict, stage_history: list) -> tuple[str, list]:
    if not gate_stages_cfg:
        return current_stage, stage_history
    stage_order = [s['name'] for s in gate_stages_cfg if 'name' in s]
    if current_stage not in stage_order:
        return current_stage, stage_history
    idx = stage_order.index(current_stage)
    diag = state.get('eil:gate:diagnostic') or {}
    survivor_density = float(diag.get('survivor_density', 0.0))
    median_trades = float(diag.get('median_trades', 0.0))
    req_density = float(stage_escalation_cfg.get('survivor_density_min', 0.05))
    req_trades = float(stage_escalation_cfg.get('median_trades_min', 20))
    patience = int(stage_escalation_cfg.get('patience_cycles', 3))
    meet = (survivor_density >= req_density) and (median_trades >= req_trades)
    if meet:
        stage_history.append({'stage': current_stage, 'cycle': cycle_idx, 'survivor_density': survivor_density, 'median_trades': median_trades})
        recent = [h for h in stage_history if h['stage'] == current_stage]
        if len(recent) >= patience and idx < len(stage_order)-1:
            old_stage = current_stage
            current_stage = stage_order[idx+1]
            logger.success(f"EIL_STAGE_ESCALATE from={old_stage} to={current_stage} cycle={cycle_idx} density={survivor_density:.3f} trades={median_trades:.1f}")
            state.set('eil:gate:stage', current_stage)
            stage_history.append({'stage': current_stage, 'cycle': cycle_idx, 'transition': True})
    state.set('eil:gate:history', stage_history)
    return current_stage, stage_history

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
        # avail = _maybe_restrict_single_symbol(cfg, avail)
        logger.info("Skipping single-symbol restriction for full panel operation.")
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

def _record_rejection(state: StateManager, reason: str, fid: str) -> None:
    rc = state.get('eil:rej:counts') or {}
    rc[reason] = int(rc.get(reason,0)) + 1
    state.set('eil:rej:counts', rc)
    # sample FIDs per reason (bounded)
    sample_key = f'eil:rej:samples:{reason}'
    samples = state.get(sample_key) or []
    if len(samples) < 25:
        samples.append(fid)
        state.set(sample_key, samples)

@click.group()
def cli(): pass

@cli.command()
def run():  # Plain function for programmatic/test use
    run_eil()

def run_eil(cfg: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    print("GEMINI_DEBUG: Entering run_eil")
    """Execute one or more Evolutionary Iteration Loop (EIL) cycles.

    This is the programmatic (non-Click) entrypoint used by tests to avoid Click
    argument parsing side-effects (pytest -q etc.). The Click CLI command `run`
    delegates to this function.

    Args:
        cfg: Optional pre-loaded configuration dict. If None, load_config() used.

    PIT NOTE: Operates solely on already-harvested lakehouse data; no forward leakage.
    """
    # --- Import path hygiene guard (stale install detection) ---
    try:
        import v26meme as _vmod
        repo_root = Path(__file__).resolve().parents[2]  # project root (fullyautobob8.8)
        mod_path = Path(_vmod.__file__).resolve()
        if repo_root not in mod_path.parents:
            msg = f"IMPORT_PATH_MISMATCH module_path={mod_path} repo_root={repo_root} (editable install missing?)"
            if os.environ.get('ALLOW_IMPORT_PATH_MISMATCH') != '1':
                raise RuntimeError(msg)
            else:
                logger.warning(msg)
    except Exception as e:
        logger.error(f"IMPORT_PATH_CHECK_FAIL err={e}")
        raise

    if cfg is None:
        cfg = load_config()
    assert isinstance(cfg, dict) and 'system' in cfg and 'discovery' in cfg and 'execution' in cfg and 'eil' in cfg
    state = StateManager(cfg['system']['redis_host'], cfg['system']['redis_port'])
    if state.get('eil:rej:counts') is None:
        state.set('eil:rej:counts', {})
    start_idx = int(state.get('eil:cycle_idx') or 0)
    if os.environ.get('EIL_FRESH') == '1' and state.get('eil:population_struct') is None:
        state.set('eil:cycle_idx', 0)
        start_idx = 0
        logger.info("EIL_FRESH_RESET applied (population_struct absent)")
        # Reset continuity suppression accumulators to avoid stale long-horizon suppression
        for k in ['eil:continuity_suppress','eil:feat:active_ct','eil:feat:cycles_ct','eil:population_hygiene']:
            if state.get(k) is not None:
                state.set(k, {}) if k != 'eil:continuity_suppress' else state.set(k, [])
                logger.info(f"EIL_CONTINUITY_RESET key={k} cleared")

    # Gate staging config (ensure present each invocation)
    gate_stages_cfg = cfg['discovery'].get('gate_stages', []) or []
    stage_escalation_cfg = cfg['discovery'].get('gate_stage_escalation', {}) or {}
    current_stage = state.get('eil:gate:stage') or (gate_stages_cfg[0]['name'] if gate_stages_cfg else 'default')
    stage_history = state.get('eil:gate:history') or []
    state.set('eil:gate:stage', current_stage)  # persist in case missing

    lake = Lakehouse(preferred_exchange=cfg["execution"]["primary_exchange"])
    ff = FeatureFactory()
    sim = SimLab()
    
    # Validator setup
    validator_config = cfg.get('validation', {}).copy()
    # Override with discovery CV params if present
    validator_config['cv_folds'] = cfg['discovery'].get('cv_folds', validator_config.get('cv_folds', 5))
    validator_config['cv_embargo_bars'] = cfg['discovery'].get('cv_embargo_bars', validator_config.get('cv_embargo_bars', 5))
    validator_config['fdr_alpha'] = cfg['discovery'].get('fdr_alpha', validator_config.get('fdr_alpha', 0.1))
    validator = Validator(validator_config)

    prober_cfg = cfg.get('prober', {})
    feature_prober: Optional[FeatureProber] = None
    if prober_cfg.get('enabled', True):
        fees_decimal = cfg['execution']['paper_fees_bps'] / 10000.0
        slippage_decimal = cfg['execution']['paper_slippage_bps'] / 10000.0
        feature_prober = FeatureProber(
            fee_pct=fees_decimal, 
            slippage_pct=slippage_decimal, 
            perturbations=int(prober_cfg.get('perturbations',64)), 
            delta_fraction=float(prober_cfg.get('delta_fraction',0.15)),
            mdd_escalation_multiplier=float(prober_cfg.get('mdd_escalation_multiplier', 1.25))
        )

    base_features = cfg['discovery'].get("base_features", [])
    if not base_features:
        raise ValueError("`discovery.base_features` not found in config or is empty.")

    # Coverage gating parameters (avoid magic numbers by sourcing config)
    min_panel_symbols = int(cfg['discovery'].get('min_panel_symbols', 3))
    min_bars_per_symbol = int(cfg['discovery'].get('min_bars_per_symbol', 100))
    timeframe_preference = ['1h','15m','5m']  # higher frequency fallback later if higher tf unseeded

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

    # Helper: record first rejection reason per formula - MOVED TO MODULE LEVEL

        max_cycles_cfg = int(cfg['eil'].get('max_cycles', 1))
    all_survivors_from_cycle: List[Dict[str, Any]] = []
    while True:
        try:
            cycle_idx = int(state.get('eil:cycle_idx') or 0)
            rel_cycles = cycle_idx - start_idx
            if rel_cycles >= max_cycles_cfg:
                logger.info(f"EIL loop complete after {rel_cycles} cycles (max_cycles={max_cycles_cfg}).")
                break

            logger.info(f"EIL_CYCLE_START idx={cycle_idx} rel={rel_cycles} stage={current_stage}")
            # Re-evaluate timeframe & coverage each outer cycle
            tf_selected, eligible_symbols, bars_map = _select_timeframe(cfg, lake, min_bars_per_symbol, min_panel_symbols, timeframe_preference)
            if tf_selected is None:
                logger.warning(f"EIL_COVERAGE_FAIL: No timeframe met coverage gates. min_symbols={min_panel_symbols} min_bars={min_bars_per_symbol}")
                state.set('eil:diag:last_reason', f"coverage_gate unmet min_symbols={min_panel_symbols} min_bars={min_bars_per_symbol}")
                _sleep(1); state.set('eil:cycle_idx', cycle_idx + 1); continue

            panel_universe = [s for s in eligible_symbols if s.endswith('_USD_SPOT')]
            panel = random.sample(panel_universe, min(cfg['discovery']['panel_symbols'], len(panel_universe))) if panel_universe else []
            logger.info(f"EIL_PANEL_SELECT tf={tf_selected} eligible={len(eligible_symbols)} usd_spot_universe={len(panel_universe)} panel_size={len(panel)}")
            
            # Inject LLM proposals at the start of the cycle using the superior seeding method
            if cfg.get('llm', {}).get('enable', False):
                try:
                    r = getattr(state, 'r', None)
                    if r:
                        # Drain the entire queue at once for efficiency
                        proposals_raw = r.lrange('llm:proposals', 0, -1)
                        if proposals_raw:
                            r.delete('llm:proposals')
                            proposals = [json.loads(p) for p in proposals_raw]
                            injected_count = gen.seed_from_proposals(proposals)
                            state.set('eil:llm_injection:last_run', {'ts': time.time(), 'injected': injected_count})
                            if injected_count > 0:
                                logger.info(f"EIL injected {injected_count} unique proposals from LLM into population.")
                except Exception as e:
                    logger.error(f"Failed to inject LLM proposals into EIL: {e}", exc_info=True)
            
            if not eligible_symbols or not panel:
                logger.warning("EIL_PANEL_EMPTY eligible_symbols=%d panel_size=%d" % (len(eligible_symbols), len(panel))); _sleep(1); state.set('eil:cycle_idx', cycle_idx + 1); continue
            
            # Build cache
            nbars_window = int(_parse_tf_bars(tf_selected, cfg['eil']['fast_window_days']))
            df_cache, feat_samples = _build_panel_cache(panel, tf_selected, nbars_window, base_features, cfg, lake, ff)
            if not df_cache:
                logger.info("EIL_DF_CACHE_EMPTY panel=%s tf=%s" % (panel, tf_selected))
                state.set('eil:diag:last_reason', 'no_df_cache_after_feature_build')
                _sleep(1); state.set('eil:cycle_idx', cycle_idx + 1); continue
            
            # Gate features, apply hygiene, and update generator state for the cycle
            feature_stats, active_feature_mask = _gate_and_stat_features(
                feat_samples=feat_samples,
                base_features=base_features,
                cfg=cfg
            )
            state.set('eil:feature_gate_diag', {f: {'keep': active_feature_mask.get(f, False)} for f in base_features})
            state.set('eil:feature_stats', {f: {**feature_stats.get(f, {}), 'keep': active_feature_mask.get(f, False)} for f in base_features})
            
            _apply_feature_hygiene(active_feature_mask, base_features, gen, state, cfg)

            # Pass the computed stats and active features to the generator for this cycle
            gen.set_feature_stats(feature_stats)
            active_features = [f for f, is_active in active_feature_mask.items() if is_active]
            
            if not active_features:
                logger.warning("EIL_NO_ACTIVE_FEATURES - all features were filtered out. Check data quality or gating thresholds.")
                generator_features = base_features
            else:
                generator_features = active_features

            gen.set_features(generator_features)

            if not gen.population:
                gen.initialize_population(); logger.info(f"EIL_POP_INIT size={len(gen.population)}")
                try:
                    with open("initial_population.json", "w") as f:
                        json.dump(gen.population, f, indent=2)
                    logger.info("Dumped initial population to initial_population.json")
                except Exception as e:
                    logger.error(f"Failed to dump initial population: {e}")

            # Gate parameters & overrides
            fdr_alpha, dsr_min_prob, min_trades_gate = _apply_stage_overrides(state, cfg, current_stage, gate_stages_cfg)
            # --- Adaptive min_trades gate ---
            try:
                # Gather recent trade count distribution from last generation diagnostics if present
                # Fallback: estimate desired trades as function of panel size * bars window * activation rate proxy
                gen_trade_samples = []
                # Look back a few prior generation diagnostics (current cycle only) for trade stats
                for gback in range(5):
                    gdiag = state.get(f'eil:diag:gen:{cycle_idx}:{gback}') or {}
                    tm = gdiag.get('trades_median')
                    if isinstance(tm, (int, float)) and tm is not None:
                        gen_trade_samples.append(float(tm))
                if gen_trade_samples:
                    median_recent = float(np.median(gen_trade_samples)) if hasattr(np, 'median') else sorted(gen_trade_samples)[len(gen_trade_samples)//2]
                else:
                    # Heuristic: expect at least one trade per symbol per 2% of bars-window (rough sparse baseline)
                    bars_window = int(_parse_tf_bars(tf_selected, cfg['eil']['fast_window_days']))
                    est = max(3, int(len(panel) * bars_window * 0.0002))
                    median_recent = float(est)
                # Scale baseline gate: require at most 1.25 * median_recent but not below configured min
                adaptive_gate = max( int(cfg['discovery'].get('promotion_criteria', {}).get('min_trades', 10)), int(median_recent * 0.75) )
                # Clamp to reasonable ceiling to avoid early starvation
                adaptive_gate = max(5, min(adaptive_gate, int(median_recent * 1.25) + 5))
                # Take the minimum between stage override gate and adaptive (adaptive should only loosen, not tighten excessively)
                eff_gate = min_trades_gate if min_trades_gate <= adaptive_gate else adaptive_gate
                if eff_gate != min_trades_gate:
                    logger.info(f"EIL_ADAPT_MIN_TRADES old={min_trades_gate} adaptive={adaptive_gate} eff={eff_gate}")
                min_trades_gate = eff_gate
            except Exception as _e:
                logger.warning(f"EIL_ADAPT_MIN_TRADES_FAIL err={_e}")
            validator.fdr_alpha = fdr_alpha
            validator.dsr_min_prob = dsr_min_prob
            validator.min_trades_gate = min_trades_gate

            generations_per_cycle = int(cfg['discovery'].get('generations_per_cycle', 1))
            
            # --- Run Generations ---
            cycle_survivors = _run_generations(
                generations_per_cycle=generations_per_cycle,
                cycle_idx=cycle_idx,
                current_stage=current_stage,
                gen=gen,
                sim=sim,
                df_cache=df_cache,
                validator=validator,
                feature_prober=feature_prober,
                state=state,
                base_features=base_features,
                panel_universe=panel,
                timeframe=tf_selected,
                cfg=cfg
            )

            # Process the collected survivors from all generations in the cycle
            if cycle_survivors:
                # Deduplicate survivors based on formula hash (alpha_id)
                unique_survivors_map = {s['alpha_id']: s for s in cycle_survivors}
                unique_survivors = list(unique_survivors_map.values())
                logger.success(f"EIL_CYCLE_SURVIVORS_FOUND total_unique={len(unique_survivors)} from cycle={cycle_idx}")
                for s in unique_survivors:
                    logger.info(f"SURVIVOR alpha_id={s.get('alpha_id','n/a')[:8]} sharpe={s.get('sharpe',-1):.2f} pval={s.get('p_value',-1):.3f} dsr={s.get('dsr_prob',-1):.2f} robust={s.get('robust_score',-1):.2f} trades={s.get('n_trades',-1)} formula={json.dumps(s.get('formula'))}")

                # Persist top-K survivors from the cycle for promotion
                survivor_top_k = int(cfg['eil']['survivor_top_k'])
                sorted_survivors = sorted(list(unique_survivors), key=lambda x: x.get('sharpe', 0.0), reverse=True)
                top_k_survivors = sorted_survivors[:survivor_top_k]
                
                all_survivors_from_cycle.extend(top_k_survivors)

                # Update state with diagnostics and candidates
                state.set('eil:survivors:last_cycle', top_k_survivors)
                for survivor in top_k_survivors:
                    alpha_id = survivor['alpha_id']
                    candidate_payload = {
                        'score': survivor.get('sharpe', 0.0),
                        'p_value': survivor.get('p_value'),
                        'dsr_prob': survivor.get('dsr_prob'),
                        'trades': survivor.get('n_trades'),
                        'formula': survivor.get('formula'),
                    }
                    state.set(f'eil:candidates:{alpha_id}', candidate_payload, ttl=48*3600) # 48h TTL
            else:
                logger.info(f"No survivors found in cycle {cycle_idx}.")


            # Stage escalation logic
            current_stage, stage_history = _maybe_escalate_stage(state, cycle_idx, current_stage, gate_stages_cfg, stage_escalation_cfg, stage_history)

            # Persist population for warm start
            state.set('eil:population_struct', gen.population)
            state.set('eil:cycle_idx', cycle_idx + 1)
            _sleep(1)

        except Exception as e:
            logger.error(f"EIL_CYCLE_FAIL idx={state.get('eil:cycle_idx') or 0} err={e}")
            import traceback
            logger.error(traceback.format_exc())
            state.set('eil:cycle_idx', (state.get('eil:cycle_idx') or 0) + 1)
            _sleep(5)
            continue
    
    return all_survivors_from_cycle

def _build_panel_cache(panel: List[str], tf_selected: str, nbars_window: int, base_features: List[str], cfg: dict, lake: Lakehouse, ff: FeatureFactory) -> Tuple[Dict[str, pd.DataFrame], Dict[str, list]]:
    """Build panel dataframe cache & empirical feature stats (quantiles) once per cycle."""
    logger.info("--- Building EIL Panel Cache & Data Quality Analysis ---")
    
    all_symbol_dfs = []
    for canon in panel:
        raw_df = lake.get_data(canon, tf_selected).tail(nbars_window)
        if raw_df.empty:
            continue
        raw_df['item'] = canon
        all_symbol_dfs.append(raw_df)

    if not all_symbol_dfs:
        return {}, {}
        
    panel_df = pd.concat(all_symbol_dfs).reset_index().set_index(['item', 'timestamp'])
    
    # The feature factory expects a multi-index dataframe and will handle per-item calculations.
    dff = ff.create(panel_df, feature_configs=cfg.get('features', {}))

    if dff.empty:
        logger.warning(f"[DATA_QUALITY] DataFrame for panel is empty AFTER feature creation.")
        return {}, {}

    logger.info(f"dff columns: {dff.columns.tolist()}")

    # The feature factory now returns a single dataframe with a multi-index.
    # We can group by item to get the individual dataframes for the cache.
    df_cache = {str(item): group for item, group in dff.groupby(level='item')}

    feat_samples: Dict[str, list] = {f: [] for f in base_features}
    for f in base_features:
        if f in dff.columns:
            col = dff[f].dropna()
            if not col.empty:
                step = max(1, len(col) // 50)
                feat_samples[f].extend(col.iloc[::step].tolist())
                
    return df_cache, feat_samples

def _gate_and_stat_features(feat_samples: Dict[str, list], base_features: List[str], cfg: dict) -> Tuple[Dict[str, Dict[str, float]], Dict[str, bool]]:
    """Calculate empirical feature stats and create an active feature mask based on quality gates."""
    feature_stats: Dict[str, Dict[str, float]] = {}
    min_non_nan = float(cfg['discovery'].get('feature_min_non_nan_ratio', 0.30))
    min_var = float(cfg['discovery'].get('feature_min_variance', 1e-6))
    active_feature_mask: Dict[str, bool] = {}
    
    logger.info("--- EIL Feature Gate Diagnostics ---")
    
    for f in base_features:
        vals = feat_samples.get(f)
        if not vals:
            active_feature_mask[f] = False
            logger.info(f"  - Feature: {f:<20} | Active: False | Reason: No values sampled.")
            continue
        
        s = pd.to_numeric(pd.Series(vals), errors='coerce').dropna()
        if s.empty:
            active_feature_mask[f] = False
            logger.info(f"  - Feature: {f:<20} | Active: False | Reason: No numeric values after coercion.")
            continue
        
        nn_ratio = float(s.notna().mean())
        
        try:
            var_raw = s.var(ddof=1) if s.notna().sum() > 1 else 0.0
            if isinstance(var_raw, complex):
                var_val = var_raw.real if pd.notna(var_raw.real) else 0.0
            elif isinstance(var_raw, (int, float)):
                var_val = float(var_raw) if pd.notna(var_raw) else 0.0
            else:
                var_val = 0.0
        except Exception as e:
            var_val = 0.0
            logger.warning(f"Could not calculate variance for {f}: {e}")

        keep = (nn_ratio >= min_non_nan) and (var_val >= min_var)
        active_feature_mask[f] = keep
        
        logger.info(f"  - Feature: {f:<20} | Active: {str(keep):<5} | Non-NaN: {nn_ratio:8.2%} (min: {min_non_nan:.2%}) | Variance: {var_val:12.6f} (min: {min_var:.6f})")

        if keep:
            feature_stats[f] = {
                'min': float(s.min()),
                'q10': float(s.quantile(0.10)),
                'q25': float(s.quantile(0.25)),
                'median': float(s.median()),
                'q75': float(s.quantile(0.75)),
                'q90': float(s.quantile(0.90)),
                'max': float(s.max())
            }
    return feature_stats, active_feature_mask

def _apply_feature_hygiene(active_feature_mask: Dict[str, bool], base_features: List[str], gen: GeneticGenerator, state: StateManager, cfg: dict) -> set:
    """Apply continuity-based suppression and reseed population if needed."""
    suppressed = {f for f, k in active_feature_mask.items() if not k}
    cont_key_active = 'eil:feat:active_ct'
    cont_key_cycles = 'eil:feat:cycles_ct'
    cont_active = state.get(cont_key_active) or {}
    cont_cycles = state.get(cont_key_cycles) or {}
    for f in base_features:
        cont_cycles[f] = int(cont_cycles.get(f, 0)) + 1
        if active_feature_mask.get(f, False):
            cont_active[f] = int(cont_active.get(f, 0)) + 1
    state.set(cont_key_active, cont_active)
    state.set(cont_key_cycles, cont_cycles)
    
    continuity_threshold = float(cfg['discovery'].get('adaptive', {}).get('continuity_threshold', 0.80))
    patience_c = int(cfg['discovery'].get('adaptive', {}).get('continuity_suppression_patience', 5))
    continuity_suppress = set()
    continuity_ratios = {}
    for f in base_features:
        a = cont_active.get(f,0); c = cont_cycles.get(f,0); ratio = a/c if c>0 else 0.0
        continuity_ratios[f] = ratio
        if c >= patience_c and ratio < continuity_threshold:
            continuity_suppress.add(f)
            
    if continuity_suppress:
        state.set('eil:continuity_suppress', list(continuity_suppress))
        
    suppressed |= continuity_suppress
    
    if suppressed:
        affected_idx = []
        for i, form in enumerate(gen.population):
            if set(_extract_features(form)) & suppressed:
                affected_idx.append(i)
        if affected_idx:
            reseed_fraction = float(cfg['discovery'].get('reseed_fraction', 0.30))
            n_reseed = max(1, int(len(affected_idx) * reseed_fraction))
            for idx in affected_idx[:n_reseed]:
                gen.population[idx] = gen._create_random_formula()
            state.set('eil:population_hygiene', {'suppressed': list(suppressed), 'continuity_suppress': list(continuity_suppress), 'affected': len(affected_idx), 'reseeded': n_reseed})
            
    state.set('eil:feature_continuity', continuity_ratios)
    return suppressed

def _run_generations(
    generations_per_cycle: int,
    cycle_idx: int,
    current_stage: str,
    gen: GeneticGenerator,
    sim: SimLab,
    df_cache: Dict[str, pd.DataFrame],
    validator: Validator,
    feature_prober: Optional[FeatureProber],
    state: StateManager,
    base_features: List[str],
    panel_universe: List[str],
    timeframe: str,
    cfg: dict
) -> List[Dict[str, Any]]:
    """Run the evolutionary algorithm for a set number of generations.

    Added diagnostics:
      - Per generation histogram + percentiles of sharpe & trades.
      - Logs top-K and bottom-K formulas by sharpe with fid + size.
      - Rejection attribution (min_trades vs p_value vs dsr) stored in Redis.
    """
    
    cycle_survivors = []
    # REMOVED: inspect hack to get feature_prober from outer scope

    for gidx in range(generations_per_cycle):
        logger.info(f"EIL_GEN_START cycle={cycle_idx} gen={gidx} pop={len(gen.population)} stage={current_stage}")
        
        evaluated_formulas = []
        zero_trade_formulas = 0
        per_formula_meta = {}

        for f in gen.population:
            fid = hashlib.sha256(json.dumps(f, separators=(',',':')).encode()).hexdigest()
            
            all_trades = []
            # The new simlab takes the full panel, not per-item dfs
            fees_pct = cfg['execution']['paper_fees_bps'] / 10000.0
            slippage_pct = cfg['execution']['paper_slippage_bps'] / 10000.0
            sim_result = sim.run_backtest(pd.concat(df_cache.values()), f, fee_pct=fees_pct, slippage_pct=slippage_pct)
            
            if sim_result is not None and not sim_result.empty:
                all_trades.append(sim_result)
            
            if not all_trades:
                zero_trade_formulas += 1
                evaluated_formulas.append({'alpha_id': fid, 'formula': f, 'trades': pd.DataFrame(), 'sharpe': -1.0})
                continue

            panel_trades_df = pd.concat(all_trades).sort_index()

            if not panel_trades_df.empty:
                sharpe = panel_trades_df['pnl'].mean() / panel_trades_df['pnl'].std() if panel_trades_df['pnl'].std() != 0 else 0
                meta = {'fid': fid, 'trades': len(panel_trades_df), 'sharpe': sharpe}
                per_formula_meta[fid] = meta
                evaluated_formulas.append({
                    'alpha_id': fid,
                    'formula': f,
                    'trades': panel_trades_df,
                    'sharpe': sharpe,
                })
            else:
                per_formula_meta[fid] = {'fid': fid, 'trades': 0, 'sharpe': -1.0}
                evaluated_formulas.append({'alpha_id': fid, 'formula': f, 'trades': pd.DataFrame(), 'sharpe': -1.0})

        # After evaluation, compute diagnostics
        if per_formula_meta:
            import numpy as _np
            trades_arr = _np.array([m['trades'] for m in per_formula_meta.values()])
            sharpe_arr = _np.array([m['sharpe'] for m in per_formula_meta.values()])
            def _pct(a,p):
                return float(_np.percentile(a, p)) if a.size else 0.0
            diag = {
                'cycle': cycle_idx,
                'gen': gidx,
                'pop': len(gen.population),
                'zero_trade_pct': round((zero_trade_formulas / max(1,len(gen.population))),4),
                'trades_median': _pct(trades_arr,50),
                'trades_p90': _pct(trades_arr,90),
                'sharpe_median': _pct(sharpe_arr,50),
                'sharpe_p90': _pct(sharpe_arr,90),
            }
            state.set(f'eil:diag:gen:{cycle_idx}:{gidx}', diag, ttl=24*3600)
            if sharpe_arr.size:
                # log concise line
                logger.info(f"EIL_GEN_DIAG cycle={cycle_idx} gen={gidx} pop={len(gen.population)} zero_trade_pct={diag['zero_trade_pct']:.2f} trades_med={diag['trades_median']} sharpe_med={diag['sharpe_median']:.2f} sharpe_p90={diag['sharpe_p90']:.2f}")
            # top/bottom sample
            sorted_by_sharpe = sorted(per_formula_meta.values(), key=lambda m: m['sharpe'])
            sample_k = min(3, len(sorted_by_sharpe))
            top_sample = sorted_by_sharpe[-sample_k:]
            bot_sample = sorted_by_sharpe[:sample_k]
            state.set(f'eil:diag:gen_top:{cycle_idx}:{gidx}', top_sample, ttl=24*3600)
            state.set(f'eil:diag:gen_bot:{cycle_idx}:{gidx}', bot_sample, ttl=24*3600)

        # Degeneracy detection
        deg_pct = (zero_trade_formulas / max(1, len(gen.population)))
        degen_threshold = min(0.85, 0.50 + 0.02 * (gen.population_size / max(1, len(base_features))))
        state.set('eil:degenerate_pct', round(deg_pct, 4))
        if deg_pct >= degen_threshold:
            logger.warning(f"EIL_DEGENERATE pct={deg_pct:.2f} >= {degen_threshold:.2f} reseeding")
            gen.initialize_population()
            continue

        # Validate the entire batch of evaluated formulas
        validated_survivors, rejections = validator.validate_batch(evaluated_formulas)

        # --- Rejection Attribution (Validation) ---
        total_rejected = 0
        rejection_summary = {}
        for reason, fids in rejections.items():
            if fids:
                # Use a more specific reason that includes the stage
                staged_reason = f"{reason}_stage_{current_stage}"
                for fid in fids:
                    _record_rejection(state, staged_reason, fid)
                total_rejected += len(fids)
                rejection_summary[reason] = len(fids)
        
        if total_rejected > 0:
            logger.info(f"EIL_GATE_REJECT stage=validation rejected={total_rejected} passed={len(validated_survivors)} breakdown={rejection_summary}")

        # Robustness filtering
        robust_survivors = []
        if feature_prober:
            robust_min = float(cfg.get('prober', {}).get('min_robust_score', 0.55))
            for survivor in validated_survivors:
                # Use the first symbol's df for robustness probe
                df_features = list(df_cache.values())[0] if df_cache else None
                if df_features is not None:
                    probe_res = feature_prober.score(df_features, survivor['formula'])
                    if probe_res.get('robust_score', 0.0) >= robust_min:
                        survivor['robust_score'] = probe_res['robust_score']
                        robust_survivors.append(survivor)
                    else:
                        # --- Rejection Attribution (Robustness) ---
                        reason = f"robust_stage_{current_stage}"
                        _record_rejection(state, reason, survivor['alpha_id'])
                else:
                    # If no data for probing, pass survivor through
                    robust_survivors.append(survivor)
            
            rejected_robust_fids_count = len(validated_survivors) - len(robust_survivors)
            if rejected_robust_fids_count > 0:
                logger.info(f"EIL_GATE_REJECT stage=robustness rejected={rejected_robust_fids_count} passed={len(robust_survivors)}")
        else:
            robust_survivors = validated_survivors # Skip if prober disabled
        
        generation_survivors = robust_survivors
        
        # Add universe and timeframe to each survivor of the generation
        for s in generation_survivors:
            s['universe'] = panel_universe
            s['timeframe'] = timeframe

        # Add survivors to the cycle list
        cycle_survivors.extend(generation_survivors)

        # Evolve population based on survivors
        fitness_scores = {item['alpha_id']: item.get('sharpe', -1.0) for item in evaluated_formulas}
        gen.run_evolution_cycle(fitness_scores=fitness_scores)

    return cycle_survivors

@cli.command(name="main")
@click.option("--config", default="configs/config.yaml", help="Path to config.yaml")
def main_command(config):
    """Main entrypoint for the EIL service."""
    cfg = load_config(config)
    run_eil(cfg)

if __name__ == "__main__":
    cli()
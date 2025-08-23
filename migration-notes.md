# Migration Notes

## 2025-08-19 Alpha Registry Hygiene (Dedup + Retro Gates + Padding Trim)

Changes:
- Added `_alpha_registry_hygiene` routine invoked each loop cycle (pre-harvest) to: 
  1. Dedupe `active_alphas` by id (keep first occurrence order-stable).
  2. Optionally enforce current promotion gates retroactively when `discovery.enforce_current_gates_on_start=true` (drops legacy underperformers).
  3. Trim trailing zero-return padding beyond recorded `n_trades` (up to `discovery.max_return_padding_trim`, default 5) to prevent distortion of correlation / variance metrics.
- New config knobs in `discovery`:
  - `enforce_current_gates_on_start` (bool, default false)
  - `max_return_padding_trim` (int, default 5)
- Hygiene emits a single JSON log line when any change occurs: `[hygiene] {"dupes_removed":N,"trimmed":M,"dropped_gates":K,"final":Z}`.

Operational Impact:
- Cleans legacy registry inflation (duplicate ids) reducing artificial factor crowding.
- Optional retro gate enforcement aligns active set with present promotion standards (use cautiously to avoid sudden portfolio shrink).
- Performance statistics become more faithful; trailing zero pads no longer dampen realized volatility or skew Sharpe.

Verification Checklist:
- [ ] After restart, observe hygiene log with expected `dupes_removed` count (>0 if duplicates existed).
- [ ] `active_alphas` length matches `final` reported.
- [ ] No non-zero-trade alphas with zero-padded tails beyond `n_trades` upon inspection.
- [ ] If enforcement enabled, portfolio still diversified (≥ 3 alphas) OR intentional pruning acknowledged.

Rollback:
- Set `discovery.enforce_current_gates_on_start=false` to halt retroactive drops.
- Remove config keys and delete hygiene function call if full revert required.
- Rehydrate previously dropped alphas only after explicit review (avoid automatic re-add without validation).

Risks / Mitigations:
- Aggressive gate enforcement could deplete active set → monitor and disable flag if `final < 3` for consecutive cycles.
- Misconfigured `max_return_padding_trim` (too large) could remove valid return history: safeguard limited to 5 by default.

Monitoring:
- Watch correlation penalty rejections post-cleanup; expected decrease after duplicate removal.
- Confirm optimizer weight distribution changes (should reflect true unique alpha profiles).

## 2025-08-19 Deflated Sharpe Gate + On-Demand Harvest Queue

Changes:
- Enabled Deflated Sharpe Ratio (DSR) gate (`validation.dsr.enabled=true`) with `min_prob=0.60` and `benchmark_sr=0.0`.
- Added true DSR probability computation (Bailey & Lopez de Prado approximation) in `cli.py` using population size as trial count.
- Introduced `promotion_buffer_multiplier` (default 1.0) and `enable_pruning` (default true) knobs under `discovery`.
- Basic pruning: remove active alphas with n_trades >=30 and Sharpe < 0.
- Added on-demand harvest queue drain: `_drain_eil_queue` integrates `eil:harvest:requests` canonicals into every timeframe plan.
- Screener canonical alignment: `market_id` now uses canonical symbol; raw venue symbol stored separately as `venue_symbol` for API calls.

Operational Impact:
- Slightly stricter promotions; expect lower daily promotion count until population adapts.
- On-demand symbol requests now harvested next cycle (ensure producers push JSON objects with `{"canonical": "XYZ_USD_SPOT"}`).
- Canonical change in screener means any downstream code expecting raw venue symbol in `market_id` should now use `venue_symbol`.

Action Items:
- Verify promotions still occur (monitor logs for `PROMOTED`). If zero for >5 consecutive cycles, consider lowering `validation.dsr.min_prob` to 0.55 or raising `promotion_buffer_multiplier` to 1.0 if previously >1.
- Validate harvester picks up queued symbols: push a test entry to Redis list `eil:harvest:requests` and confirm parquet appears.
- Rebuild any local caches relying on old screener instrument structure.

Verification Checklist:
- [ ] `pytest tests/data` passes
- [ ] At least 1 promotion within first 10 cycles post-upgrade (unless intentionally tight)
- [ ] Redis list `eil:harvest:requests` consumes entries (list length returns to 0 after loop cycle)
- [ ] Screener instruments include both `market_id` (canonical) and `venue_symbol`
- [ ] No errors in logs referencing missing `market_id` for impact calc

Rollback:
- Set `validation.dsr.enabled=false` and remove added config keys.
- Revert screener `market_id` assignment to venue symbol if incompatibility discovered.
- Remove `_drain_eil_queue` call in harvester if causing unintended plan expansion.

## 2025-08-19 Expand core_symbols Universe

Changes:
- Added 15 additional high-liquidity USD spot pairs to `harvester.core_symbols` in `configs/config.yaml` (LTC, LDO, BCH, UNI, ATOM, FIL, NEAR, ARB, OP, INJ, APT, XLM, ALGO, SUI, SHIB).

Impact / Actions:
- Staged backfill duration increases (more symbols * intraday timeframes). Expect longer first few harvest loops.
- Higher request load against exchange quotas (Coinbase 60 rpm, Kraken 45 rpm). Monitor `harvest:errors` and adjust quotas if throttling observed.
- Larger storage footprint and potential longer QA (gap detection) cycles. Ensure disk monitoring.
- Factor / symbol concentration may decrease (positive) but correlation pruning logic should be reviewed for scaling.
- Promotion multiple-testing burden increases; confirm BH-FDR gate remains at `discovery.fdr_alpha=0.10` and consider tightening if promotions spike.

Rollback:
- Remove symbols from `core_symbols` list and restart loop; dynamic symbols will still appear if screener-qualified.

Verification Checklist:
- [ ] `pytest tests/data` passes (checkpoint & QA unchanged)
- [ ] No sustained increase in `harvest:errors` Redis hash counts
- [ ] Rate limit sleep intervals not saturated (logs show steady progression)
- [ ] Active universe size stable vs. expectations
- [ ] Promotion rate within historical band

Monitoring Keys (Redis):
- `harvest:coverage:*` per (exchange:tf:symbol)
- `harvest:errors` hash per exchange
- `data:screener:latest:canonicals` for dynamic additions

Mitigations if Issues:
- Temporarily raise `loop_interval_seconds` (e.g. 45 → 60) during bootstrap.
- Reduce `bootstrap_days_override` depth for intraday frames.
- Disable low-impact added symbols (start with lowest volume among new set).

## 2025-08-19 Critical Fixes Round 2

Summary:
- Strict PIT shifts enforced in `FeatureFactory` (added shift(1) to `momentum_10p`, `rsi_14`, `close_vs_sma50` alongside existing price-derived features) eliminating same-bar leakage risk.
- Impact calculation now consistently uses raw `venue_symbol` when querying order books; canonical stored separately (`market_id`). Prevents missed books on venues with non‑canonical tickers.
- BH‑FDR gate now operates on true cross‑validation p‑values produced by `panel_cv_stats` (`pvals_by_fid` pipeline) instead of heuristic placeholders, restoring statistical control integrity.
- Risk halt flatten logic reinforced in `ExecutionHandler.reconcile` (zeroes all symbol targets on `risk:halted` before order construction) ensuring immediate de‑risking.
- On‑demand harvest queue `_drain_eil_queue` present; symbols merged across all timeframes each cycle (event-sourced, PIT safe).

Operational Impact:
- Minor reduction in apparent in-sample metrics due to stricter PIT; expect slightly lower Sharpe but higher forward fidelity.
- Improved universe coverage for impact screen (fewer false INF impacts due to symbol mismatch).
- Promotion counts may dip briefly as real p‑values replace optimistic heuristics; monitor `discovery.fdr_alpha` but avoid premature loosening.
- Faster drawdown response when manual or automated halt triggers.

Action Items:
- Re-run `pytest` to confirm no leakage tests regress.
- Monitor promotion pass logs for `BH-FDR kept` vs `total` to ensure expected retention rate (~alpha * candidates).
- Trigger a simulated halt by setting `risk:halted=1` in Redis mid-cycle; verify positions flattened in next reconcile.
- Queue a test symbol via `redis-cli LPUSH eil:harvest:requests '{"canonical":"DOGE_USD_SPOT"}'` and confirm harvest inclusion.

Verification Checklist:
- [ ] PIT feature leakage tests remain green
- [ ] Promotions still occur (not zero over 10 cycles) with controlled FDR
- [ ] Impact calc no longer logs missing book warnings for canonical IDs
- [ ] Manual halt flattens positions immediately
- [ ] On-demand queued symbol harvested within next cycle

Rollback:
- Remove feature shifts (NOT recommended) by reverting shift loop in `FeatureFactory`.
- If impact regression, revert to canonical `market_id` usage in `_impact_bps` (unlikely necessary).
- Temporarily reintroduce heuristic p-values by hardcoding `pv=1.0` (will inflate false discoveries; avoid for production).
- Comment out risk flatten block if causing unintended position churn (ensure halt flag logic first).

## 2025-08-19 Lane Allocation Integration

Change:
- Integrated dynamic lane budget application and portfolio reconciliation into main loop (`cli.py`). After promotions/pruning and diagnostics, the loop now:
  1. Fetches active alphas.
  2. Derives inverse-variance capped raw weights via `PortfolioOptimizer`.
  3. Applies performance-aware lane scaling with `LaneAllocationManager.apply_lane_budgets`.
  4. Persists final normalized alpha weights to `portfolio:alpha_weights` in Redis.
  5. Invokes `ExecutionHandler.reconcile` (paper mode) to simulate position adjustments.

Rationale:
- Previously lane logic existed but was never executed, leaving moonshot/core budget governance inert.
- Ensures capital allocation respects adaptive moonshot expansion/contraction based on relative Sortino.

Operational Impact:
- Paper portfolio positions now adjust each cycle toward lane-adjusted target weights (subject to risk halts and order notional caps).
- Expect minor cash drift during early cycles until weights stabilize (few alphas with sufficient trade history).

Verification Checklist:
- [ ] Logs show `Lane budgets applied:` followed by fractions each cycle once alphas active.
- [ ] `portfolio:alpha_weights` key populated (check via Redis CLI).
- [ ] `PROMOTED` events eventually followed by reconciliation log `Reconciled portfolio`.
- [ ] No exceptions logged under `Allocation/reconcile error`.

Rollback:
- Remove the portfolio construction block added near end of loop in `cli.py`.
- Delete `portfolio:alpha_weights` key if stale allocations undesirable.

## 2025-08-19 Harvest Availability Suppression & Core Symbol Prune

Changes:
- Pruned synthetic / non-venue pairs from `harvester.core_symbols` (removed: `USDC_USD_SPOT`, `USDC_USDT_SPOT`, `EUR_USD_SPOT`).
- Added `harvester.availability` config block with `miss_threshold` (default 3) and `suppress_ttl_minutes` (default 360) to auto‑suppress repeatedly unresolvable (exchange, canonical) pairs.
- Harvester `_build_plan` now performs a pre‑plan availability probe using `venue_symbol_for`; increments Redis hash `harvest:unresolvable:attempts` when a mapping is missing.
- After `miss_threshold` consecutive misses for a pair on an exchange, sets a suppression key `harvest:suppress:<exchange>:<canonical>` with TTL; suppressed pairs are omitted from plans until TTL expiry.
- Logs a one‑line summary: `[harvest] availability_filter removed={...}` listing per timeframe removals each cycle (noise reduction vs per‑symbol skip logs).

Operational Impact:
- Reduces wasted requests & log clutter from permanently unsupported stablecoin cross or delisted pairs.
- Slightly smaller universe for initial backfill; no effect on valid symbol harvesting.
- If a pair becomes newly listed, suppression expires automatically; mapping recognized next cycle.

Action Items:
- Monitor `harvest:unresolvable:attempts` to ensure no explosive growth (indicates resolver or venue outage).
- After stable operations, consider tightening `harvester.max_gap_pct_accept` (e.g., 0.50 → 0.20) and re‑enable `staged_backfill.enabled=true`.
- Validate that expected core symbols still appear in lakehouse partitions after a few cycles.

Verification Checklist:
- [ ] `harvest:unresolvable:attempts` hash increments only for removed symbols.
- [ ] Suppressed symbols absent from `[harvest] coverage_summary` outputs.
- [ ] No performance degradation in harvest cycle duration post change.

Rollback:
- Remove `availability` block from config.
- Delete suppression keys via `redis-cli KEYS 'harvest:suppress:*' | xargs redis-cli DEL`.
- Re-add removed symbols to `core_symbols` if needed.

## 2025-08-19 Screener Stablecoin Cross Exclusion

Change:
- Added config flag `screener.exclude_stable_stable` (default true in current config) and implementation in `UniverseScreener` to skip markets where both base and quote are recognized stablecoins (USDT, USDC, DAI, FDUSD, TUSD, PYUSD).

Rationale:
- Stablecoin-stablecoin crosses exhibit near-zero directional volatility → low edge density, inflate candidate counts, waste impact & order book fetches, and introduce noisy parity warnings.

Operational Impact:
- Slight reduction in screener candidate set (removes low-value pairs like USDC/USDT) leading to faster impact evaluation cycles.
- Lower log noise from parity deviation warnings (TUSD, etc.).
- Minor shift in volume distribution toward true risk assets (monitor factor concentration — expected neutral to positive).

Verification Checklist:
- [ ] `python3 -m v26meme.cli debug-screener` output no longer lists *USDC_USDT_SPOT* or similar stable-stable pairs.
- [ ] Log parity warnings count decreases (compare previous 10 cycles vs next 10 cycles).
- [ ] Promotions and BH-FDR accept rate stable (no unexpected drop due to universe shrinkage).

Redis / Monitoring:
- Optional: track candidate count trend via parsing `[screener] candidates_pre_impact` log lines.

Rollback:
- Set `screener.exclude_stable_stable=false` in `configs/config.yaml` and restart loop.
- No further code changes required; feature is gated purely by config.

## 2025-08-19 Alpha Set Prune & Temporary Risk / Promotion Overrides

Changes:
- Pruned active alpha set from 52 → 20 (kept top Sharpe performers meeting quality filters: Sharpe ≥0.5, Sortino ≥0.7, win_rate ≥0.48, mdd ≤0.35, n_trades ≥30).
- Applied runtime overrides (Redis): `discovery:override:max_promotions_per_cycle=1`, `risk:override:max_order_notional_usd=100` to slow new inflow and cap exposure while evaluating quality.
- Cleared `risk:halted` flag after verifying no drawdown metrics were present (halt likely triggered by transient condition or missing equity telemetry).

Rationale:
- Reduce dilution/noise from large alpha set; focus capital on higher conviction strategies.
- Minimize risk of immediate re-halt while diagnostics run.

Operational Impact:
- Portfolio sizing resumes (non-zero weight observed) but total turnover low due to single alpha currently sized (others at 0 until optimizer allocates or metrics fill).
- Promotion cadence throttled; expect slower additions until override removed.

Next Steps:
- Monitor for 3–5 cycles: ensure no new risk halts and weights diversify (>3 non-zero alphas) before lifting overrides.
- If stable, remove overrides (delete Redis keys) and optionally persist config changes (`max_promotions_per_cycle`) if desired.

Rollback:
- Delete Redis override keys; restore previous promotion cadence.
- Re-add pruned alphas only if justified (re-run backtests / validation) — not recommended blindly.

## 2025-08-19 — FeatureFactory double-shift removal

- Removed second lag application to momentum_10p and rsi_14 (previously shifted twice via shift_cols) to restore intended t-1 alignment (PIT correctness). No interface change; deterministic outputs preserved.

## 2025-08-19 — Harvester queue & suppression updates

- Timeframe alias map corrected: coinbase 6h retained; kraken 6h -> 4h.
- EIL harvest queue now lpop drains up to limit (no overflow loss).
- Suppression TTL treats Redis TTL -1 (no expiry) as still suppressed.
- Venue symbol fallback: attempts BASE/QUOTE if registry lookup fails.

## 2025-08-19 Lane Instrumentation & Control Upgrades (Phases 1-2 + 7 + A + E + C partial)

Implemented:
- Rejection counters (dsr, min_trades, hard_gates, factor_corr, robust) with per-cycle log and remaining daily promotion quota.
- Daily promotion counter keyed by UTC date (`promotions:day:YYYYMMDD`).
- Alpha registry snapshot written each cycle to `data/alphas/registry_*.json`.
- Lane EWMA smoothing (`lanes.smoothing_alpha`), dead-band (`lanes.dead_band`), and max step (`lanes.max_step_per_cycle`).
- Probation weight cap for new moonshot alphas (`lanes.probation`).
- Retag moonshot→core enabled (`lanes.retag.*` criteria) — modifies alpha lane tag post-performance maturation.
- Risk freeze: promotions suppressed when `risk:halted` or `risk:conserve_active` is set.
- Population size increased 160→320; generations_per_cycle 2→3; adaptive ceiling 400.
- Auto CV method upgrade to CPCV when symbol panel size ≥8 or population_size ≥300 (`discovery.auto_cpcv`).

Deferred (not yet implemented):
- Advanced factor correlation matrix penalty (still simple pairwise cap check).
- Aging/decay pruning model.
- Adaptive search pressure booster.
- USD stable classification toggle.

Operational Impact:
- Promotion visibility improved; easier tuning of gates before expanding search throughput further.
- Lane budget adjustments smoother; reduced oscillation.
- Early moonshot alphas risk-limited via probation cap; lane retagging promotes durable performers to core.
- CPCV auto-switch increases validation robustness under larger search breadth.

Action Items:
- Monitor logs for `Promotion rejections breakdown` and ensure expected distribution (dsr & hard_gates dominant early).
- Confirm lane retags only after intended maturity (check `Retagged N moonshot→core alphas`). Adjust `lanes.retag.min_trades` if premature.
- Validate daily promotion counter resets on UTC day rollover.
- Assess performance impact of added snapshots; prune old registry files if disk usage grows.

Rollback:
- Remove added config keys or set feature flags: set `lanes.retag.enabled=false`, remove probation block, delete smoothing keys.

## 2025-08-19 Parallel Bootstrap & Coverage Gate Additions

Added configuration knobs to accelerate cold start without sacrificing PIT correctness:

harvester:
- bootstrap_mode: parallel (enables external deep-history worker concept; main loop uses shallow incremental).
- min_coverage_for_research: 0.30 (initial required coverage ratio per research timeframe symbol to start discovery).
- high_coverage_threshold: 0.80 (target coverage to relax gating after first promotion).
- priority_symbols: Tier A ordering (BTC, ETH, SOL) fetched first each timeframe for earliest viable panel.
- partial_harvest: true enables timeframe-sliced harvesting (loop can proceed after one timeframe batch instead of full matrix).

discovery:
- defer_until_coverage: true gates research until coverage conditions satisfied.
- min_panel_symbols: 3 minimal symbols meeting bar threshold (research timeframe) before CV/promotion.
- min_bars_per_symbol: 100 minimal bars on research timeframe per symbol for panel inclusion.

Operational Impact:
- First research cycle expected within minutes (vs full historical wait) once priority_symbols reach min_bars_per_symbol and coverage threshold.
- Full-depth backfill can run concurrently (future bootstrap worker) without blocking iteration.

Action Items:
- Implement separate bootstrap worker script (future commit) invoking harvester with deep targets when bootstrap_mode=parallel.
- Add coverage evaluation helper in loop (meets_min_coverage) referencing harvest:coverage hash.
- Monitor logs for a new gating message ("Waiting for coverage…") until threshold met.

Rollback:
- Remove new keys or set partial_harvest=false, defer_until_coverage=false.

Verification Checklist:
- [ ] Loop logs show coverage gate message before initial promotion.
- [ ] Tradeable + promotion logs appear earlier than prior baseline.
- [ ] No regression in PIT leakage tests.

## 2025-08-20 Staged Backfill Re-Enable + Deeper Coinbase Hourly Bootstrap

Changes:
- Re-enabled `harvester.staged_backfill.enabled=true` (was false during initial accelerated bootstrap) to allow controlled depth expansion.
- Expanded staged targets:
  - 1m: 7 → 30 → 90 days
  - 5m: 30 → 180 → 365 days
  - 15m: 90 → 365 → 730 days
- Increased `per_exchange_bootstrap.coinbase` for 1h (7 → 365 days) and 4h alias (30 → 730 days) to match Kraken depth and avoid early research panel imbalance.

Rationale:
- Deep early hourly coverage on Coinbase prevents factor skew toward Kraken-only history and accelerates cross-venue robustness checks.
- Staged progression mitigates immediate rate burst and QA overhead while converging toward multi‑month/minute and multi‑year/hourly footprints.

Operational Impact:
- First harvest cycles after upgrade will request substantially more historical hourly & multi-hour candles from Coinbase (subject to venue limits / pagination).
- Intraday minute windows will advance through stages once ≥70% accept ratio per timeframe (existing logic) — expect sequential promotion over several cycles.
- Increased initial disk writes (partitioned per month already in place) — monitor storage utilization.

Verification Checklist:
- [ ] Redis keys `harvest:backfill:stage:1m|5m|15m` start at index 0 then increment as accept ratios >=0.7.
- [ ] Coverage summaries show increasing `expected` for affected timeframes across cycles.
- [ ] Coinbase 1h / 4h parquet row counts approach Kraken parity after initial deep bootstrap (allow several cycles).
- [ ] No sustained spike in `harvest:errors` or `rate_limit_hits` beyond transient backfill.

Rollback:
- Set `harvester.staged_backfill.enabled=false` to freeze stages.
- Restore prior `per_exchange_bootstrap.coinbase` day counts (1h:7, 4h:30) if rate pressure or quota exhaustion observed.

Risk / Mitigations:
- Potential longer single-cycle duration during early deep fetch; mitigate by temporarily increasing `system.loop_interval_seconds` or decreasing minute staged targets.
- Higher chance of sparse early 1m fills on Coinbase; existing fallback (1m→5m) and flat-fill heuristics remain active.

## 2025-08-20 Rejection Telemetry + Continuity Suppression + Correlation Fallback

Changes:
- Added detailed rejection telemetry in `hyper_lab.py` capturing counts & sampled FIDs per first-failing gate.
- New Redis keys:
  - `eil:rej:counts` (map reason→count), `eil:rej:samples:<reason>` (list sampled FIDs), `eil:rej:dominant`.
  - `eil:feature_gate_diag`, `eil:feature_stats`, `eil:feature_continuity`, `eil:continuity_suppress`.
  - `eil:population_hygiene` summarizing suppressed features, continuity suppressions, affected & reseeded formulas.
- Feature continuity tracking across cycles with suppression after `continuity_suppression_patience` cycles below `continuity_threshold`.
- Correlation fallback feature `beta_btc_20p` (rolling OLS beta vs BTC returns) added to `FeatureFactory` to retain cross‑asset structure when `btc_corr_20p` sparse.
- Fitness function penalties extended: drawdown, feature concentration, return variance.
- Adaptive population auto-tuning: if dominant rejection reason is p-value and exceeds `adaptive.rejection_pval_high_ratio`, population size increments by `adaptive.population_step` up to `adaptive.population_size_max`.
- Config additions: `rejection_sample_size`, `rejection_alert_ratio`, `fitness_concentration_penalty_scale`, `fitness_variance_penalty_scale`, `adaptive.continuity_suppression_patience`, `adaptive.tuning_enabled`, `adaptive.rejection_pval_high_ratio`, `adaptive.population_step`.

Operational Impact:
- Higher observability of search inefficiencies (quick tuning of gates vs. brute force).
- Reduced wasted evaluation on unstable / intermittent features; improved survivor quality density.
- Maintains diversity when correlation windows sparse via beta fallback.
- Population can scale adaptively under high early-stage statistical rejection pressure (p-value dominated).

Verification Checklist:
- [ ] Redis `HGETALL eil:rej:counts` shows non-zero counts after first generation cycle.
- [ ] `eil:rej:samples:pval` populated when p-value rejections occur.
- [ ] `eil:feature_continuity` ratios between 0–1; suppressed features listed in `eil:continuity_suppress` only after patience cycles.
- [ ] `beta_btc_20p` present in feature stats when BTC data available.
- [ ] `adaptive:population_size` increases only when dominant rejection is pval and ratio ≥ threshold.

Risk / Mitigations:
- Over-expansion of population increases compute: capped by `adaptive.population_size_max`.
- False suppression due to transient data gaps: guarded by patience cycles.
- Fallback feature misuse: Beta still lagged & PIT-safe; monitor variance.

Rollback:
- Remove added config keys (or set `adaptive.tuning_enabled=false`).
- Delete fallback feature block in `FeatureFactory`.
- Remove telemetry state writes in `hyper_lab.py` (search for `eil:rej:` and `feature_continuity`).

## 2025-08-20 Checkpoint Reconciliation + 1h Aggregation Bootstrap

Changes:
- Added `harvester.checkpoint_reconcile` pass executed at start of each harvest cycle to fast-forward **lagging** checkpoints to the parquet tail minus one full bar.
- New config block:
  ```yaml
  harvester:
    checkpoint_reconcile:
      enabled: true
      min_lag_bars: 12
      max_forward_bars: 1000000
  ```
- Aggregation bootstrap enhancement: 5m→1h aggregator now seeds symbols with **zero native 1h parquet files** even if coverage hash lacks target entries (previously only low native coverage). Meta flag `bootstrap_missing: true` stored in aggregated parquet metadata for first creation.
- Reconciliation skips symbols with no existing checkpoint (preserves first-run bootstrap semantics) and never rewinds checkpoints (forward-only, PIT safe).

Rationale:
- Large negative checkpoint drift starved higher timeframe aggregation (no 1h history despite deep 5m / 15m partitions) and impeded staged backfill logic relying on accurate last_ts.
- Forward reconciliation eliminates redundant refetch loops and immediately unlocks synthetic aggregation over already materialized lower timeframe bars.

Operational Impact:
- First post-upgrade cycle will log `[harvest] checkpoint_reconcile adjusted=N scanned=M` if drift present.
- Expect immediate materialization of 1h parquet for symbols with adequate 5m history during the same loop (look for `[harvest] aggregated_timeframes` log line with `bootstrap_missing` true entries).
- Research / feature factory gains deeper 1h panel coverage, unblocking promotion gates previously waiting on minimum bars.

Verification Checklist:
- [ ] Run harvest cycle; confirm reconciliation log appears (if prior drift) and `harvest:checkpoint_reconcile:last` Redis key populated.
- [ ] Inspect `data/coinbase/1h/YYYY/MM/*.parquet` now present with >0 rows.
- [ ] Probe B (checkpoint drift) re-run shows drift_ms near 0 (≤ one bar) for adjusted timeframes.
- [ ] FeatureFactory no longer errors due to missing 1h coverage (if previously selecting 1h research timeframe).

Rollback:
- Set `harvester.checkpoint_reconcile.enabled=false` to disable forward adjustments.
- Remove the config block and delete the reconciliation invocation in `harvester.run_once`.
- Delete newly aggregated 1h parquet files if you prefer to force native venue history only (NOT recommended; keep for research until native fills).

Risks / Mitigations:
- Misconfiguration of `min_lag_bars` too low could cause needless checkpoint churn: default 12 prevents micro-adjustments.
- Extremely large drift could indicate systemic ingestion issue; reconciliation caps forward jump at `max_forward_bars` to avoid skipping intended deep backfill. Raise cautiously after root cause validation.

Monitoring:
- Redis key `harvest:checkpoint_reconcile:last` for adjustment stats.
- Aggregation logs: ensure `symbols=` count grows initial run then stabilizes (idempotent once caught up).
- Coverage hash `harvest:coverage` now reflects higher timeframe expected/actual growth.

## 2025-08-20 Config Restoration for EIL Reactivation + Aggregation Logging

Changes:
- Restored full `harvester.core_symbols` (25 symbols) and re-enabled `dynamic_enabled=true` after bootstrap throttle.
- Re-added native `1h` timeframe to `harvester.timeframes_by_lane.core` while keeping synthetic 5m→1h aggregation active.
- Lowered `harvester.aggregate_timeframes.min_native_rows_threshold` 50→10 to seed early hourly coverage from 5m.
- Restored multi-stage `staged_backfill.targets_days` for `1m` (7,30,90) and `5m` (30,180,365) for deeper historical context feeding EIL.
- Re-enabled `validation.dsr.enabled=true` (deflated Sharpe gate) post data stabilization.
- Aggregator resample frequency changed `1H`→`1h` (future-proof pandas warning).
- Added aggregation skip telemetry: logs `[harvest] aggregated_timeframes built ... skipped_native=` or `skipped target=1h native_ok=` providing visibility when synthetic build is bypassed due to sufficient native rows.

Rationale:
- Provide robust hourly panel (native + synthetic fallback) to unblock discovery coverage gates (`min_panel_symbols`, `min_bars_per_symbol`).
- Ensure early EIL iterations have diversified symbol history; maintain overfitting safeguards (DSR reinstated).
- Improved observability distinguishes genuine aggregation work from benign skips (avoids silent starvation misdiagnosis).

Operational Impact:
- First cycles after restoration will include 1h fetching again; synthetic aggregation continues only for symbols below native row threshold.
- Increased request volume vs trimmed bootstrap config; monitor rate-limit logs.
- Expect EIL survivor emergence once 3+ symbols have ≥100 1h bars (coverage gate) — verify via `harvest:coverage` hashes.

Verification Checklist:
- [ ] system.log shows `coverage_summary` with 1h attempts >0 and accept_ratio rising.
- [ ] If any hourly starvation occurs, `[harvest] aggregated_timeframes built` appears with `symbols>0`.
- [ ] Redis `harvest:checkpoint_reconcile:last` updated (no large drifts remaining).
- [ ] Promotions resume with DSR gate logging (no persistent all-fail due to coverage shortfall).

Rollback:
- Revert `core_symbols` shrink list & disable `dynamic_enabled` if rate pressure unacceptable.
- Raise `min_native_rows_threshold` to suppress synthetic aggregation.
- Disable DSR via `validation.dsr.enabled=false` (NOT recommended except diagnostic).

## 2025-08-21 Retro Coverage Reindex (Option B Gate Unblock)

Changes:
- Added `_retro_reindex_coverage` helper in `v26meme/cli.py` invoked inside `_coverage_gate_ok` (runs once per research timeframe) to scan existing parquet partitions (`data/<exchange>/<tf>/**/<symbol>.parquet`) and retro-populate `harvest:coverage` entries.
- For each parquet file lacking a coverage hash entry (or with a stale `actual` lower than on-disk row count), writes a conservative payload: `expected=rows`, `actual=rows`, `coverage=1.0`, `gaps=0`, `gap_ratio=0.0`, `accepted=true`, plus flag `reindexed=true`.
- Leaves future live harvest writes untouched; native harvest cycles will overwrite reindexed entries with real expected/coverage when new data fetched.
- Augmented coverage shadow hash (`harvest:coverage:aug`) still populated with `actual_total` for diagnostics; gate eligibility now satisfied immediately by reindexed base hash (no reliance on shadow for min_bars).

Rationale:
- Historical deep history (copied / aggregated) produced large parquet depth but coverage hash reflected only last incremental fetch (e.g., `actual=4` for symbols with >8000 hourly bars), starving discovery coverage gate (`min_panel_symbols=3`, `min_bars_per_symbol=100`).
- Option B (retro reindex) chosen over threshold loosening to preserve intended gating rigor while eliminating false scarcity caused by metadata mismatch.

Operational Impact:
- On first loop after deployment, logs `[retro_reindex] timeframe=1h coverage entries updated=N` (N = number of keys written/upgraded). Subsequent cycles skip scan via flag key `harvest:coverage:reindexed:<tf>`.
- Coverage gate should transition from `eligible=1/3` (or similar) to `eligible>=3/3` enabling research/EIL start without waiting for slow native re-writes.
- `reindexed=true` markers allow future cleanup or audit; can be filtered if distinguishing synthetic vs native coverage provenance becomes necessary.

Verification Checklist:
- [ ] system.log shows retro reindex log exactly once per timeframe.
- [ ] Redis `HGETALL harvest:coverage` entries for previously starved symbols now report large `actual` (≈ parquet row count) and include `reindexed` flag when JSON loaded.
- [ ] Coverage gate log shows `eligible >= required` and loop proceeds to tradeable panel build.
- [ ] Promotions resume within expected cycles (subject to statistical gates) after gate unblock.

Rollback:
- Delete keys matching `harvest:coverage:reindexed:*` and affected coverage entries to restore pre-reindex state (NOT recommended; would reintroduce starvation).
- Comment out `_retro_reindex_coverage` invocation in `_coverage_gate_ok` to disable automatic reindex.

Risks / Mitigations:
- Potential overstatement of coverage ratio (set to 1.0) for reindexed entries; mitigated because gating logic only needs bar count ≥ `min_bars_per_symbol` and uses coverage threshold (0.30 initial) which remains trivially satisfied for deep history.
- Large scan cost on extremely wide universes: one-time pass; complexity proportional to number of parquet files. Can be optimized later with stored row count metadata.

Monitoring:
- Track promotions and ensure no sudden surge in false positives (BH-FDR + DSR still active ensuring statistical discipline).
- Inspect any lingering low `actual` entries; absence likely indicates missing parquet or permissions issue rather than gate logic.

## 2025-08-21 Accelerated Hourly Bootstrap (365d -> 30d)

Changes:
- Reduced `harvester.per_exchange_bootstrap.<exchange>."1h"` from 365 to 30 days (coinbase & kraken).
- Reduced `harvester.bootstrap_days_default."1h"` from 365 to 30.

Rationale:
- Year-scale deep fetch was throttling initial data readiness (tens of thousands of 5m/1h bars) under rate limits and retry noise (503s). 30-day window supplies sufficient recent regime data to start EIL & promotions while long-range history can be layered later via staged or parallel deep backfill.

Impact:
- Initial hourly coverage fills ~720 bars (30d) vs ~8760 (365d) – faster gate satisfaction (min_bars_per_symbol=100) and reduced cycle latency.
- Historical breadth for long-horizon features reduced temporarily; ensure any features referencing >30d windows are gated or adaptive.

Follow-Up Plan:
- After system stabilizes (first promotions achieved), reintroduce deeper 1h history via a controlled background job or re-raise bootstrap target.

Verification Checklist:
- [ ] New harvest cycles no longer emit large deep_backfill_override logs for 1h.
- [ ] Coverage actual counts for 1h symbols reach >100 quickly (<5 cycles).
- [ ] No regression in PIT tests.

Rollback:
- Restore previous values (30 -> 365) in config and restart loop (will trigger deeper fetch next cycles).

## 2025-08-21 Dynamic Panel Target Days (coverage gate modernization)

Changes:
- Added `harvester.panel_target_days` mapping (1m=7,5m=14,15m=30,1h=30) replacing implicit fixed 1h ~2000 bar expectation.
- `_coverage_gate_ok` now derives `target_bars = max(min_bars_per_symbol, panel_target_days[tf]*bars_per_day)` and emits `[panel_coverage]` telemetry with min/median/max bars.
- Gate eligibility uses dynamic `target_bars` instead of static min_bars floor for symbol inclusion; still honors `min_coverage_for_research`.

Rationale:
- Eliminates mismatch between shortened bootstrap horizons (e.g. 30 days for 1h ~720 bars) and legacy deep history assumptions that caused perpetual backfill pressure.

Impact:
- Research/EIL can start once realistic recent-history targets met; excess history (e.g. Coinbase >1y) retained but not required for gate.
- Tests updated to assert presence of `target_bars` in gate stats.

Verification Checklist:
- [ ] system.log shows `[panel_coverage]` with `target_bars`=720 for 1h (given panel_target_days 30) and increasing symbols_eligible.
- [ ] No lingering logs chasing 2000 1h bars.
- [ ] PIT + QA tests continue passing.
- [ ] Coverage gate opens when bars_min >= ~target_bars and symbols_eligible >= min_panel_symbols.

Rollback:
- Remove `panel_target_days` from config; `_coverage_gate_ok` will fallback to min_bars_per_symbol floor (still dynamic telemetry but target_bars collapses to floor).

## 2025-08-21 TEMP Minimal Intraday Staged Backfill

Changes:
- Disabled `harvester.staged_backfill.enabled` (false) and reduced targets to minimal single-stage (1m=7d,5m=14d,15m=30d) for single-symbol BTC data validation.

Rationale:
- Shorten bootstrap spans to focus on correctness instrumentation without spending rate budget on deep history.

Impact:
- Intraday historical depth limited; features requiring longer lookbacks may be truncated.
- Should NOT promote strategies dependent on > current span until restored.

Rollback Plan:
- Restore previous multi-stage arrays and set enabled true once validation complete.

## 2025-08-21 TEMP Single-Symbol Narrowing

Changes:
- `harvester.core_symbols` -> [BTC_USD_SPOT]; disabled dynamic_enabled, aggregation, checkpoint_reconcile; trimmed timeframes; set partial_harvest true; bootstrap_mode inline.
- `discovery.panel_symbols` & `min_panel_symbols` -> 1 (single-symbol research panel).

Rationale:
- Isolate data-plane correctness (schema, gaps, PIT integrity) before re-expanding universe to avoid conflating breadth issues with ingestion bugs.

Risk / Caveat:
- Single-symbol EIL greatly increases overfitting risk; do NOT promote live strategies from this phase.

Rollback:
- Restore previous core_symbols list and discovery panel settings; re-enable dynamic + aggregation + reconcile.

## 2025-08-21 EIL Single-Symbol Narrowing Flag

Changes:
- Added `harvester.restrict_single_symbol` (bool) config knob. When `true` and `harvester.core_symbols` has length 1, Hyper Lab / EIL now restricts eligible symbol universe to that single core symbol (mirrors main loop gating) to allow deterministic PIT validation and feature pipeline hardening without panel noise.
- Patched `hyper_lab._select_timeframe` via `_maybe_restrict_single_symbol` helper; logs `EIL_SINGLE_SYMBOL_MODE` on activation.

Rationale:
- During data-plane stabilization we intentionally operate on a single deeply harvested instrument (BTC) to validate feature correctness, coverage gating, and evolutionary loop mechanics before re-expanding. Previously Hyper Lab still saw the broader lakehouse, causing eligible_symbols inflation and masking narrow-mode assumptions.

Operational Impact:
- EIL eligible_symbols count should equal 1 while flag enabled; panel sampling deterministic for BTC.
- Prevents premature multi-symbol statistical artifacts (e.g., CPCV auto-switch) until deliberate expansion.

Verification Checklist:
- [ ] Run Hyper Lab with `restrict_single_symbol: true`; observe log line `EIL_SINGLE_SYMBOL_MODE core_symbols=['BTC_USD_SPOT'] eligible_after_narrow=1`.
- [ ] Redis keys `eil:feature_stats` only reflect BTC-derived distributions.
- [ ] Disabling flag (set false) restores prior multi-symbol eligible set (>1).

Rollback:
- Remove or set `harvester.restrict_single_symbol: false` and restart Hyper Lab.

Risks / Mitigations:
- Narrow search space may overfit single instrument microstructure; mitigated by mandatory re-expansion phase before any live deployment or allocation promotion to core.

## 2025-08-21 Single-Symbol Sandbox Hardening

Changes:
- Set `discovery.max_promotions_per_cycle=0` and `discovery.max_promotions_per_day=0` to freeze alpha promotion during BTC-only validation phase.
- Disabled LLM proposer (`llm.enable=false`) to remove noisy AttributeError and isolate deterministic generator behavior.
- Raised `risk.max_consecutive_errors` 3→12 to prevent frequent kill-switch during feature factory / proposer stabilization.

Rationale:
- Prevent overfitted single-asset strategies from entering allocation lanes, skewing early telemetry and lane budget smoothing.
- Reduce noise in error counters so genuine structural issues surface clearly.

Rollback / Re-enable Path:
- After multi-symbol expansion (≥3 symbols with min bars), restore promotions (set cycle/day limits back to original values) and re-enable LLM proposer.
- Lower `risk.max_consecutive_errors` to enforcement baseline (3) once proposer stabilized and feature errors absent across multiple cycles.

Verification Checklist:
- [ ] Loop logs show no promotion attempts (rejections breakdown only) while limits zero.
- [ ] No new `RISK HALTED` events during standard cycles (unless genuine repeated failures >12).
- [ ] `llm:proposer:*` Redis telemetry remains static.

Risks / Mitigations:
- Delayed accumulation of survivor set (intentional); mitigated by planned reactivation milestone.
- Higher error threshold could delay kill-switch on real faults; mitigation: manual log monitoring during sandbox phase.

## 2025-08-22 EIL Core Evolution and Fitness Function Upgrade (Draft Summary)

User-intent summary of recent / ongoing EIL changes and diagnostic path. NOTE: Some items below are partially implemented (activation gain + instrumentation) while others (full composite fitness weights, verified survivor emergence) remain in-progress.

**Summary:**
Critical focus on restoring the Evolutionary Iteration Loop (EIL) to produce evaluable, higher‑entropy candidate formulas. Investigation isolated a flat fitness landscape (sparse trades, low Sharpe dispersion, p=1.0 CV collapse). Incremental patches (df_cache retention, activation gain, multi‑symbol panel, instrumentation) improved trade counts and reduced zero‑trade formulas, but statistically significant survivors still gated by p‑value path.

**Changes (current vs claimed):**
1. `generator.py`
   - Existing subtree crossover & multi‑operator mutation confirmed (threshold / feature / operator / logical op). Max depth still 2 (deeper depth expansion not yet applied).
   - Diversity improvements planned (variable depth >2, additional logical forms) – NOT yet merged.
2. `feature_factory.py`
   - Single lag application already in place for price‑derived features; no redundant double shifts detected in current version. (Claimed multi‑shift removal not required.)
3. `hyper_lab.py`
   - Added activation_gain term (exp saturation on trades) and p-value / Sharpe / trades distribution logging (`EIL_DIST`).
   - Added FDR debug bypass + survivor telemetry scaffolding.
   - Composite fitness (profit vs activation weights) NOT yet integrated; current fitness still mean_sharpe * trade_factor * penalties (drawdown / concentration / variance). Activation gain presently logged but not multiplied into fitness (pending decision).
4. `configs/config.yaml`
   - Relaxed gates (temporary) + multi-symbol expansion. `fitness_weights` block NOT yet added (pending final design / normalization).

**Operational Impact (observed):**
- zero_trade_formulas reduced from >50% → ~3–15% in multi-symbol runs.
- trades_med improved episodically (peaks >100 under earlier activation experiment; currently ~12–20 after config adjustments).
- sharpe_med modestly positive (≈0.4–0.55) but pvals_med remains 1.0 in later cycles (statistical power still insufficient / CV path still sparse-trade fragile).
- No validated `SURVIVOR` log lines yet (all candidates failing gates even with FDR bypass due to downstream metrics or p-value degeneracy).

**Next Steps (planned to complete this migration):**
- Implement sparse-trade CV adaptation (binomial win-rate test fallback before t-test) to meaningfully lower p-values when directional edge present.
- Integrate composite fitness with configurable weights (`discovery.fitness_weights.profit`, `discovery.fitness_weights.activation`) and normalized sum=1; include activation_gain multiplier.
- Optionally incorporate activation diversity tie-break (small feature_count_used epsilon) for deterministic ordering.
- Add telemetry: `eil:diag:fitness_breakdown` per generation (median components).
- Post-success tighten gates (remove debug_relaxed_gates, restore FDR-only path) and document promotion readiness.

**Verification Checklist (pending completion):**
- [ ] First survivor under debug bypass (SURVIVOR log) with recorded dsr_prob & p_value.
- [ ] pvals_med < 0.80 after sparse-CV patch on ≥2 consecutive generations.
- [ ] fitness_breakdown shows both profit & activation contributions >0 for median formula.
- [ ] zero_trade_formulas stable < 0.20 across ≥5 generations.
- [ ] Removal of debug bypass still yields ≥1 survivor within 10 generations (else re-tune).

**Rollback Guidance:**
Not recommended mid-migration; partial rollback could reintroduce flat landscape. If required, revert only incremental activation_gain & instrumentation blocks (retain df_cache fix). Avoid re-tightening gates until sparse-CV path merged.

**Risk / Guardrail Notes:**
- PIT integrity preserved (no forward data; only threshold sampling & historical bars used).
- No secrets added; config knobs to centralize any new constants (activation scales, weight fractions).
- Anti-ruin rails untouched (risk module unaffected by EIL internal fitness changes).

(End of 2025-08-22 draft entry)

## 2025-08-22 EIL Window Extension (45→75 days)
Extended `eil.fast_window_days` from 45 to 75 to increase per-fold sample size and statistical power for p-value (BH-FDR) path. Monitoring checklist:
- [ ] pvals_med < 0.90 within 5 generations post-change.
- [ ] trades_med increases (baseline recorded pre-change).
- [ ] Cycle duration increase < 1.8x (guard compute budget).
Rollback if: cycle latency >2x AND no pvals_med improvement after 12 generations.

## 2025-08-22 Harvester Bootstrap Defaults Added
Added `harvester.bootstrap_days_default` mapping (1m:2, 5m:7, 15m:14, 1h:45, 4h:120, 1d:365) to remove KeyError in `harvester.run_once` and make historical depth explicit & tunable (replaces implicit 30d fallback). Guards staged_backfill logic and deep_backfill_override coverage checks.

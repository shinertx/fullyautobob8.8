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

Monitoring:
- `harvest:coverage` hash for per-symbol depth.
- `harvest:rate_limit_hits` for throttling.
- Disk: monthly partitions under `data/coinbase/{1h,6h}/YYYY/MM/`.

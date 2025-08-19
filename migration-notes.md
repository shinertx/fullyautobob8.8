# Migration Notes

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

Change: Added 15 additional high-liquidity USD spot pairs to `harvester.core_symbols` in `configs/config.yaml` (LTC, LDO, BCH, UNI, ATOM, FIL, NEAR, ARB, OP, INJ, APT, XLM, ALGO, SUI, SHIB).

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

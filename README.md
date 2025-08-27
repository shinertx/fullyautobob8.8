# 🧠 v26meme v4.7.5

**Autonomous AI Quant — From $200 → $1,000,000 in 30 Days**  
Paper-first, live-ready. Self-improving. PIT-correct. Ruin-proof.

---

## 🚀 Project Mission

v26meme is a fully autonomous, **LLM-driven trading intelligence** engineered from first principles.  
It evolves like a quant hedge fund in a box:

* **Discover** thousands of causal edges (order flow, liquidity, incentives, volatility regimes).
* **Validate** rigorously with PIT-correct features, purged CV, BH-FDR false-discovery controls, and robustness probing.
* **Allocate** dynamically with Kelly-scaled risk management and factor-aware diversification.
* **Compound** capital hyper-aggressively, yet obey anti-ruin guardrails.

**Scoreboard:** $200 → $1M in 30 days (≈ 5,000% monthly).  
**Reality:** Continuous, exponential edge discovery and capital compounding.

---

## 📂 File Tree

```
fullyautobob8.8/
├── configs/
│   ├── config.yaml
├── v26meme/
│   ├── __init__.py
│   ├── cli.py
│   ├── core/
│   │   ├── dsl.py
│   │   └── state.py
│   ├── data/
│   │   ├── harvester.py
│   │   ├── lakehouse.py
│   │   ├── quality.py
│   │   ├── checkpoints.py
│   │   ├── universe_screener.py
│   │   ├── screener_store.py
│   │   ├── token_bucket.py
│   │   ├── top_gainers.py
│   │   ├── usd_fx.py
│   │   ├── asset_registry.py
│   │   └── maintenance.py
│   ├── registry/
│   │   ├── canonical.py
│   │   ├── catalog.py
│   │   ├── resolver.py
│   │   └── venues.py
│   ├── research/
│   │   ├── feature_factory.py
│   │   ├── generator.py
│   │   ├── validation.py
│   │   └── feature_prober.py
│   ├── labs/
│   │   ├── hyper_lab.py
│   │   ├── simlab.py
│   │   └── screener_replay.py
│   ├── allocation/
│   │   ├── optimizer.py
│   │   └── lanes.py
│   ├── execution/
│   │   ├── exchange.py
│   │   ├── handler.py
│   │   ├── risk.py
│   │   └── micro_live.py
│   ├── llm/
│   │   └── proposer.py
│   └── analytics/
│       └── adaptive.py
├── dashboard/
│   ├── app.py
│   └── hyper_lab_app.py
├── tests/
│   ├── data/
│   ├── research/
│   ├── labs/
│   ├── llm/
│   ├── registry/
│   └── execution/
├── .github/
│   ├── copilot-instructions.md
│   └── prompts/
├── data_harvester.py
├── install_and_launch_v475.sh
├── migration-notes.md
├── requirements.txt
├── pyproject.toml
├── README.md
└── pytest.ini
```

> See Copilot instructions for high‑impact role of each module (Prime Directive alignment).

---

## 🏗️ Architecture

* **Screener → Lakehouse → FeatureFactory → Generator/EIL → CV+FDR → Promotion → Optimizer → Risk → Execution (paper/live) → Dashboard**
* **Event-sourced harvester**: Lane-aware timeframes, resumable sync, quality gates, canonical joins.
* **Extreme Iteration Layer (Hyper-Lab)**: Run 5k+ hypotheses/loop, publish survivors.
* **Ensemble Intelligence**: A "meta-alpha" is created by combining the signals from the top-performing core alphas, resulting in a more robust and stable signal.
* **Lane system**:
  * Core (95%) = stable edges
  * Moonshot (5%, dynamic) = top-gainers, resets every 2–4h
* **Risk**: Daily stops, equity floors, phase scaling, conserve mode, kill-switch.
* **LLM**: OpenAI-only, JSON-hardened suggestions, capped and budgeted.
* **Simulator**: Unified, calibrated from micro-live, venue-fee aware.

---

## ⚙️ How to Run

This project is designed to run as a persistent, multi-process application using `tmux` to manage the main trading loop and the dashboards.

### Prerequisites

Before you begin, ensure you have the following installed:
*   **Git**
*   **Python 3.10+**
*   **Redis**: Must be running and accessible. On Debian/Ubuntu, you can install and start it with:
    ```bash
    sudo apt-get update && sudo apt-get install redis-server
    sudo systemctl enable --now redis-server
    ```
*   **tmux**: A terminal multiplexer used to run background processes.
    ```bash
    sudo apt-get install tmux
    ```

### 1. Installation

First, clone the repository and navigate into the project directory:
```bash
git clone https://github.com/shinertx/fullyautobob8.8.git
cd fullyautobob8.8
```

### 2. Configuration

The system requires API keys and other secrets to be stored in a `.env` file.

1.  Copy the example file:
    ```bash
    cp .env.example .env
    ```
2.  **Edit `.env`** and add your API keys, particularly `OPENAI_API_KEY`.

### 3. Launching the System

The included script automates the setup of a Python virtual environment and launches all necessary processes within a `tmux` session.

```bash
./install_and_launch_v475.sh
```

This script will:
1.  Create a Python virtual environment at `.venv/`.
2.  Install all required dependencies from `requirements.txt`.
3.  Start four `tmux` sessions in the background:
    *   `trading_session`: Runs the main `v26meme.cli loop`, which includes the harvester, alpha discovery, and execution logic.
    *   `dashboard_session`: Serves the main Streamlit dashboard on port `8601`.
    *   `hyper_lab`: Runs the headless Extreme Iteration Layer (EIL) process.
    *   `hyper_lab_dashboard`: Serves the Hyper-Lab (EIL) dashboard on port `8610`.

### 4. Accessing the Dashboards

Once the system is running, you can access the web interfaces:

*   **Main Dashboard**: [http://localhost:8601](http://localhost:8601)
*   **Hyper Lab Dashboard**: [http://localhost:8610](http://localhost:8610)

### 5. Managing the Sessions

You can attach to the `tmux` sessions to view logs or manage the processes.

*   **List sessions**:
    ```bash
    tmux ls
    ```
*   **Attach to the main loop session**:
    ```bash
    tmux attach -t trading_session
    ```
    *(Press `Ctrl+b` then `d` to detach)*

*   **Stop all sessions and kill the bot**:
    ```bash
    tmux kill-session -t trading_session
    tmux kill-session -t dashboard_session
    tmux kill-session -t hyper_lab
    tmux kill-session -t hyper_lab_dashboard
    ```

### Manual Execution

If you prefer to run the components manually without the launch script:

1.  **Set up the environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
2.  **Run the main loop**:
    ```bash
    python3 -m v26meme.cli loop
    ```
3.  **Run the main dashboard** (in a separate terminal):
    ```bash
    streamlit run dashboard/app.py --server.port 8601
    ```
4.  **Run the Hyper Lab dashboard** (in a separate terminal):
    ```bash
    streamlit run dashboard/hyper_lab_app.py --server.port 8610
    ```

---

## 📊 Dashboard


* Equity & drawdown charts
* Portfolio allocations & active alphas
* Risk telemetry (Kelly, max order, daily stop)
* Micro-Live probe stats

---

## 🔒 Hard Rules (Engineering Guardrails)

1. **No magic numbers** — always adaptive or config-driven.
2. **PIT correctness** — no lookahead, all features lagged/shifted.
3. **Fail closed** — bad data never written; alphas never promoted without robust validation.
4. **Reproducibility** — seeds pinned, deps pinned, configs versioned.
5. **Anti-ruin first** — daily stops, equity floors, per-alpha accountability.
6. **Factor awareness** — no illusory diversification; promotion penalizes correlation.
7. **Transparency** — dashboard, logs, telemetry always on.

---

## 🔄 Recent Enhancements

+ **Alpha Registry Hygiene (2025-08-19):** Automatic per-cycle maintenance now cleans `active_alphas` by (1) removing duplicate ids (order-preserving), (2) trimming trailing zero-return padding beyond recorded `n_trades` (up to `discovery.max_return_padding_trim`, default 5) to prevent correlation / variance skew, and (3) optionally retro-enforcing current promotion gates when `discovery.enforce_current_gates_on_start=true`. A single JSON log line `[hygiene] {...}` summarizes actions (dupes_removed, trimmed, dropped_gates, final). Disable retro enforcement by leaving the flag false (default) to avoid sudden portfolio contraction.
+ **Fast Bootstrap Pipeline (2025-08-19):** Added `harvester.bootstrap_mode: parallel` (placeholder worker hook) to allow deep backfill to execute out-of-band while the main loop begins discovery on shallow data.
+ **Partial Harvest Timeframe Rotation:** `harvester.partial_harvest: true` processes only one timeframe per loop cycle (round‑robin) to reach minimal research coverage faster (e.g. fill `1h` before grinding lower granularities). State key `harvest:partial:tf_index` tracks rotation.
+ **Coverage Gate & Escalation:** Research is deferred until at least `discovery.min_panel_symbols` symbols have both (a) >= `harvester.min_coverage_for_research` coverage ratio and (b) >= `discovery.min_bars_per_symbol` bars on the research timeframe (default 1h). After the first promotion, threshold escalates to `harvester.high_coverage_threshold` (flag `coverage:raise_threshold`). Gate can be disabled via `discovery.defer_until_coverage: false`.
+ **Priority Symbols:** `harvester.priority_symbols` harvested first each cycle to seed flagship bases (BTC, ETH, SOL by default) ensuring early panel viability and adaptive knob calculation.
+ **Promotion Threshold Buffer Escalation:** First successful promotion sets a state flag raising the coverage requirement to accelerate depth-building before scaling additional alphas.
+ **Expanded Config Knobs:** Added `discovery.min_panel_symbols`, `discovery.min_bars_per_symbol`, `harvester.min_coverage_for_research`, `harvester.high_coverage_threshold`, and lane smoothing parameters (`lanes.smoothing_alpha`, `lanes.dead_band`, `lanes.max_step_per_cycle`, probation & retag blocks) — all adaptive & testable.

* **Deflated Sharpe Gate (2025-08-19):** Activated probabilistic overfit filter (`validation.dsr.enabled=true`, `min_prob=0.60`). Strategies must clear deflated Sharpe confidence threshold after BH-FDR.
* **On-Demand Harvest Queue:** `eil:harvest:requests` now drained each cycle; queued canonicals are force-included in the harvester plan (all configured timeframes) for next ingestion.
* **Screener Canonical Alignment:** Screener now sets `market_id` to canonical (e.g. `BTC_USD_SPOT`) and preserves raw venue symbol as `venue_symbol` for impact calculation and API calls; ensures downstream joins are consistent with Lakehouse.
* **Basic Alpha Pruning:** Automatic removal of stale alphas (>=30 trades & Sharpe < 0) controlled by `discovery.enable_pruning`.
* **Promotion Buffer Knob:** `promotion_buffer_multiplier` allows optional tightening (defaults 1.0 — neutral).
* **Expanded Core Universe (2025-08-19):** Added high-liquidity USD spot pairs (LTC, LDO, BCH, UNI, ATOM, FIL, NEAR, ARB, OP, INJ, APT, XLM, ALGO, SUI, SHIB) to `harvester.core_symbols`.
* **Orderflow microstructure features:** The screener now injects order-book metrics (bid/ask imbalance, spread (bps), microprice deviation) into each instrument when `feeds.orderflow.enabled=true`. This enriches universe selection with real-time liquidity signals.
* **Combinatorial Purged CV (CPCV):** In addition to standard purged K-fold cross-validation, you can enable CPCV (set `discovery.cv_method: "cpcv"`) to evaluate strategies across multiple train/test fold combinations for more robust fitness estimates.
* **Candle hygiene:** The Lakehouse data loader drops the latest OHLCV bar if it’s still in an open interval, preventing partial-bar lookahead. This check parses the timeframe (e.g. 1m, 1h) and omits any not-yet-closed candle from model data.
* **Group C (2025-08-22) Diversity & Speciation:** Added structural feature-set Jaccard clustering (`discovery.diversity.*`) with diversity fitness weight (`fitness_extras.weights.diversity`), per-cycle telemetry `eil:diag:diversity`, greedy speciation clusters, diversity bonus (1 - median Jaccard), cluster size penalty, and promotion gate requiring ≥1 (≥2 for large populations) clusters to reduce convergence and factor crowding.
* **PBO Telemetry & Robustness Gate (2025-08-23):** Added Probability of Backtest Overfitting (PBO) metric per generation (`eil:pbo:last`) using deterministic symbol splits (config: `discovery.pbo_splits`, `discovery.pbo_min_symbols`). Integrated robustness probing gate (FeatureProber) that perturb-tests strategy thresholds/operators (`prober.*` config) and rejects fragile formulas below `min_robust_score`. Introduced stagnation detector (tracks median p-value history; if improvement < `min_pval_delta` over `window` cycles, mutation rates bumped by `mutation_bump`) to escape local optima while preserving deterministic PIT correctness.

## 🌐 Data & Universe

**Orderflow Features:** When order flow feeds are enabled, the UniverseScreener enriches each instrument with real-time order book snapshots. It computes top-level imbalance, spread (in basis points), and microprice deviation via `OrderflowSnap` and attaches these as `of_imbalance`, `of_spread_bps`, `of_microprice_dev` fields. This provides the strategy generator with microstructure context (e.g. if an asset’s order book is skewed or wide) during universe selection. These features are optional and controlled by `feeds.orderflow.enabled` in the config.

**Candle Hygiene:** The Lakehouse ensures data integrity by enforcing candle completeness. In `Lakehouse.get_data()`, the final bar is dropped if its timestamp falls within the current active interval (i.e. not fully closed). The timeframe string (minutes, hours, etc.) is parsed dynamically to determine the expected bar duration, so whether using 1m or 4h bars, any partial candle (for example, a 4h candle that hasn’t finished) is excluded from the dataset. This prevents any inadvertent lookahead bias from including a bar that is still forming.

## 🛡️ Validation & Robustness

**Purged & Combinatorial CV:** Strategy validation uses Purged K-Fold cross-validation to get an unbiased estimate of out-of-sample performance. Now, users can opt for **Combinatorial Purged CV** (`discovery.cv_method: "cpcv"`) to further stress-test strategies. CPCV runs multiple overlapping fold combinations (e.g. testing two folds at a time) to simulate more rigorous train/test splits. This helps ensure a discovered edge isn’t an artifact of one lucky split.

**Deflated Sharpe Ratio Gate:** To reduce false discoveries from many trials, the promotion logic can enforce a Deflated Sharpe Ratio threshold. When enabled (`validation.dsr.enabled=true`), each candidate’s out-of-sample Sharpe significance is adjusted for the number of strategies evaluated. Candidates that don’t meet the minimum confidence (`min_prob`) that their Sharpe is real (not luck) will be rejected before promotion. This adds a secondary safeguard atop the BH-FDR control, further lowering the chance of overfit strategies entering the portfolio.

**PBO Logging:** The system now computes the **Probability of Backtest Overfitting (PBO)** for each batch of strategies and logs it in the strategy promotion loop. It does so by measuring the rank correlation between in-sample and out-of-sample strategy performance across random splits of the panel (e.g. splitting the set of symbols into design vs. test sets). A negative rank correlation in a split means strategies that ranked high in-sample ranked poorly out-of-sample – a sign of overfitting. PBO is reported as the fraction of such adverse splits, giving a sense of how likely the entire batch’s results are due to overfitting. Lower PBO (near 0%) is desirable, indicating robust performance that generalizes, while higher PBO approaching 100% flags that many strategies might not hold up out-of-sample.

---

## 🔍 EIL Telemetry & Adaptive Search (2025-08-20)

New observability & adaptation layer added to Hyper Lab (Extreme Iteration Layer):

Telemetry Keys:
- `eil:rej:counts` / `eil:rej:samples:<reason>` — first-failing gate rejection distribution (pval, dsr, trades, sortino, sharpe, win_rate, mdd) + sampled FIDs.
- `eil:feature_gate_diag`, `eil:feature_stats` — per-cycle feature gating & empirical quantiles.
- `eil:feature_continuity`, `eil:continuity_suppress` — rolling continuity ratios & features suppressed after patience threshold.
- `eil:population_hygiene` — summary of suppressed features & partial reseed actions.
- `eil:rej:dominant` — dominant rejection reason (if any) crossing alert ratio.
- `adaptive:population_size` — current auto-tuned genetic population size.

Adaptive Mechanics:
- Partial reseed of formulas referencing suppressed or low-continuity features (`discovery.reseed_fraction`).
- Fitness penalties: drawdown (`fitness_drawdown_penalty_scale`), concentration (`fitness_concentration_penalty_scale`), variance (`fitness_variance_penalty_scale`).
- Auto-tuning: if p-value rejections dominate (≥ `adaptive.rejection_pval_high_ratio`), population size increments by `adaptive.population_step` (bounded by `adaptive.population_size_max`).
- Correlation fallback feature `beta_btc_20p` (rolling lagged OLS beta vs BTC) inserted when correlation sparse; maintains cross-asset structural signal channel.

Configuration Additions:
```yaml
discovery:
  rejection_sample_size: 8
  rejection_alert_ratio: 0.70
  fitness_concentration_penalty_scale: 0.25
  fitness_variance_penalty_scale: 0.15
  reseed_fraction: 0.30
  fitness_drawdown_penalty_scale: 0.40
  adaptive:
    continuity_suppression_patience: 5
    tuning_enabled: true
    rejection_pval_high_ratio: 0.75
    population_step: 40
```

Use Cases:
- Rapidly identify dominant bottleneck (e.g. 80% p-value rejections → expand population / widen feature spans; trade count rejections → lower min_trades or extend window).
- Monitor feature stability over time; continuity < threshold triggers suppression limiting overfitting to sporadic artifacts.
- Preserve diversity & structural insight when primary correlation feature underpopulated via beta fallback.

---

## 🔁 Progressive Gate Staging (2025-08-23)

Discovery now supports **multi-stage statistical gating**:

```yaml
discovery:
  gate_stages:
    - name: relaxed   # exploratory breadth
      fdr_alpha: 0.20
      dsr_min_prob: 0.40
      min_trades: 10
    - name: normal    # consolidation
      fdr_alpha: 0.12
      dsr_min_prob: 0.55
      min_trades: 25
    - name: tight     # capital-ready rigor
      fdr_alpha: 0.08
      dsr_min_prob: 0.65
      min_trades: 40
  gate_stage_escalation:
    survivor_density_min: 0.05
    median_trades_min: 20
    patience_cycles: 3
validation:
  bootstrap:
    enabled: true
    n_iter: 500
    min_trades: 60
    seed: 1337
    method: basic
```

Mechanics:
- Each EIL cycle computes `survivor_density = accepts/population` and `median_trades` of accepted formulas.
- When both metrics exceed escalation thresholds for `patience_cycles` successive qualifying cycles, stage advances.
- Stage overrides base `fdr_alpha`, `dsr_min_prob`, and trade gate.
- Rejection telemetry keys now suffixed with stage (e.g. `pval_stage_relaxed`).

Benefits:
- Prevents early over-tight gating that starves evolutionary diversity.
- Tightens statistical rigor only after population demonstrates evaluability depth.
- Stage-aware rejection analytics accelerate diagnosis of bottleneck (p-values vs trades vs DSR).

Bootstrap Path:
- For low trade-count evaluations (`total_oos_trades < validation.bootstrap.min_trades`), a deterministic bootstrap p-value replaces parametric t-test.
- Stabilizes early-stage significance estimates, reducing false negatives from high variance tiny samples.

Operational Tips:
- If promotions halt post-escalation, relax the final stage `fdr_alpha` slightly (e.g. 0.08 → 0.10) or raise `survivor_density_min` to delay tightening.
- To debug bootstrap activation, temporarily set `validation.bootstrap.min_trades` very large (e.g. 10000) and confirm p-values persist.

---

## 🧪 Tests

Run:

```bash
pytest tests/
```

Or run the main autonomous loop directly:

```bash
python3 -m v26meme.cli loop
```

Debug the screener only:

```bash
python3 -m v26meme.cli debug_screener
```

Covers:

* Harvester checkpointing/resume
* Schema validation & QA gates
* Canonical mapping
* PIT leakage checks

---

## 🛠️ Development Flow

* Paper-first always (no accidental live trades).
* tmux sessions: `trading_session`, `dashboard_session`, `hyper_lab`.
* Logs in `logs/system.log`.
* Use `.github/copilot-instructions.md` to keep Copilot aligned with **Prime Directive**:

  > *Maximize compounding per unit risk while strictly obeying anti-ruin guardrails.*

---

## 🛣️ Roadmap → v5.0

* Add LaneAllocationManager (dynamic budgets).
* Factor model & risk-parity optimizer.
* Unified tick-level simulator.
* Meta-RL optimizer for generator parameters.
* Dynamic venue scaling (DEX connectors).

---

## Dashboards

The system includes two Streamlit dashboards for monitoring:

1.  **Main Dashboard**: Provides an overview of portfolio performance, equity curve, and active trading strategies.
    - **URL**: `http://localhost:8601`
    - **Command**: `streamlit run dashboard/app.py --server.port 8601`

2.  **Hyper Lab Dashboard**: Shows the status of the strategy discovery process (EIL), including survivor candidates and feature analysis.
    - **URL**: `http://localhost:8610`
    - **Command**: `streamlit run dashboard/hyper_lab_app.py --server.port 8610`

---

✅ **Ready to run, black-box autonomous quant.**  
📈 Compounding scoreboard = $200 → $1M in 30 days.

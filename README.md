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
v26meme/
├── configs/
│   ├── config.yaml            # System, discovery, portfolio, risk, feeds
│   └── symbols.yaml           # Canonical exchange symbol registry
├── v26meme/
│   ├── cli.py                 # Main loop (discovery → validation → promotion → execution)
│   ├── core/                  # State & DSL
│   ├── data/                  # Lakehouse, screener, registry, harvester
│   ├── feeds/                 # CryptoPanic, OnChain, Orderflow
│   ├── research/              # Feature factory, generator, validation, prober
│   ├── labs/                  # SimLab, Hyper-Lab (EIL), Screener replay
│   ├── allocation/            # Portfolio optimizer
│   ├── execution/             # Exchange, risk, handler, micro-live
│   ├── llm/                   # OpenAI proposer (hard JSON guardrails)
│   └── analytics/             # Adaptive knobs, telemetry
├── dashboard/
│   └── app.py                 # Streamlit dashboard (equity, risk, alpha tables)
├── tests/
│   └── data/                  # Harvester QA, checkpointing, canonical joins
├── install_and_launch_v475.sh # One-command installer/launcher (tmux sessions)
├── data_harvester.py          # Adaptive, event-sourced harvester
├── requirements.txt           # Pinned deps for reproducibility
├── pyproject.toml             # Metadata
├── README.md                  # (this file)
└── .github/
    ├── copilot-instructions.md # Custom Copilot guardrails
    └── prompts/                # Task-specific Copilot prompts
```

---

## 🏗️ Architecture

* **Screener → Lakehouse → FeatureFactory → Generator/EIL → CV+FDR → Promotion → Optimizer → Risk → Execution (paper/live) → Dashboard**
* **Event-sourced harvester**: Lane-aware timeframes, resumable sync, quality gates, canonical joins.
* **Extreme Iteration Layer (Hyper-Lab)**: Run 5k+ hypotheses/loop, publish survivors.
* **Lane system**:
  * Core (95%) = stable edges
  * Moonshot (5%, dynamic) = top-gainers, resets every 2–4h
* **Risk**: Daily stops, equity floors, phase scaling, conserve mode, kill-switch.
* **LLM**: OpenAI-only, JSON-hardened suggestions, capped and budgeted.
* **Simulator**: Unified, calibrated from micro-live, venue-fee aware.

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/v26meme.git
cd v26meme

# Install & launch (paper mode, default)
./install_and_launch_v475.sh
```

**Dependencies pinned** in `requirements.txt`.  
**Dashboard:** http://localhost:8601

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

* **Orderflow microstructure features:** The screener now injects order-book metrics (bid/ask imbalance, spread (bps), microprice deviation) into each instrument when `feeds.orderflow.enabled=true`. This enriches universe selection with real-time liquidity signals.
* **Combinatorial Purged CV (CPCV):** In addition to standard purged K-fold cross-validation, you can enable CPCV (set `discovery.cv_method: "cpcv"`) to evaluate strategies across multiple train/test fold combinations for more robust fitness estimates.
* **Deflated Sharpe & PBO checks:** A Deflated Sharpe Ratio gate can now filter out promoted strategies lacking statistical significance (`validation.dsr.enabled=true` with a `min_prob` confidence threshold). The system also logs the Probability of Backtest Overfitting (PBO) each cycle, based on in/out-of-sample rank correlation of strategy performance. Higher PBO indicates greater overfitting risk.
* **Candle hygiene:** The Lakehouse data loader drops the latest OHLCV bar if it’s still in an open interval, preventing partial-bar lookahead. This check parses the timeframe (e.g. 1m, 1h) and omits any not-yet-closed candle from model data.

## 🌐 Data & Universe

**Orderflow Features:** When order flow feeds are enabled, the UniverseScreener enriches each instrument with real-time order book snapshots. It computes top-level imbalance, spread (in basis points), and microprice deviation via `OrderflowSnap` and attaches these as `of_imbalance`, `of_spread_bps`, `of_microprice_dev` fields. This provides the strategy generator with microstructure context (e.g. if an asset’s order book is skewed or wide) during universe selection. These features are optional and controlled by `feeds.orderflow.enabled` in the config.

**Candle Hygiene:** The Lakehouse ensures data integrity by enforcing candle completeness. In `Lakehouse.get_data()`, the final bar is dropped if its timestamp falls within the current active interval (i.e. not fully closed). The timeframe string (minutes, hours, etc.) is parsed dynamically to determine the expected bar duration, so whether using 1m or 4h bars, any partial candle (for example, a 4h candle that hasn’t finished) is excluded from the dataset. This prevents any inadvertent lookahead bias from including a bar that is still forming.

## 🛡️ Validation & Robustness

**Purged & Combinatorial CV:** Strategy validation uses Purged K-Fold cross-validation to get an unbiased estimate of out-of-sample performance. Now, users can opt for **Combinatorial Purged CV** (`discovery.cv_method: "cpcv"`) to further stress-test strategies. CPCV runs multiple overlapping fold combinations (e.g. testing two folds at a time) to simulate more rigorous train/test splits. This helps ensure a discovered edge isn’t an artifact of one lucky split.

**Deflated Sharpe Ratio Gate:** To reduce false discoveries from many trials, the promotion logic can enforce a Deflated Sharpe Ratio threshold. When enabled (`validation.dsr.enabled=true`), each candidate’s out-of-sample Sharpe significance is adjusted for the number of strategies evaluated. Candidates that don’t meet the minimum confidence (`min_prob`) that their Sharpe is real (not luck) will be rejected before promotion. This adds a secondary safeguard atop the BH-FDR control, further lowering the chance of overfit strategies entering the portfolio.

**PBO Logging:** The system now computes the **Probability of Backtest Overfitting (PBO)** for each batch of strategies and logs it in the strategy promotion loop. It does so by measuring the rank correlation between in-sample and out-of-sample strategy performance across random splits of the panel (e.g. splitting the set of symbols into design vs. test sets). A negative rank correlation in a split means strategies that ranked high in-sample ranked poorly out-of-sample – a sign of overfitting. PBO is reported as the fraction of such adverse splits, giving a sense of how likely the entire batch’s results are due to overfitting. Lower PBO (near 0%) is desirable, indicating robust performance that generalizes, while higher PBO approaching 100% flags that many strategies might not hold up out-of-sample.

---

## 🧪 Tests

Run:

```bash
pytest tests/
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

✅ **Ready to run, black-box autonomous quant.**  
📈 Compounding scoreboard = $200 → $1M in 30 days.

fullyautobob8.8/
├─ .github/
│  ├─ copilot-instructions.md
│  └─ prompts/
├─ .vscode/
├─ configs/
│  ├─ config.yaml
│  └─ symbols.yaml
├─ dashboard/
│  └─ app.py
├─ v26meme.egg-info/
├─ v26meme/
│  ├─ cli.py                  # main loop orchestrator
│  ├─ core/                   # state & DSL
│  ├─ data/                   # lakehouse, screener, registry, harvester, QA, etc.
│  ├─ feeds/                  # cryptopanic, onchain, orderflow
│  ├─ research/               # feature factory, generator, validation, prober
│  ├─ labs/                   # simlab, hyper_lab (EIL), screener_replay
│  ├─ allocation/             # portfolio optimizer, lane budgets
│  ├─ execution/              # exchange, risk, handler, micro-live
│  ├─ llm/                    # OpenAI proposer
│  └─ analytics/              # adaptive knobs, telemetry
├─ tests/
│  └─ data/
├─ .env.example
├─ .gitignore
├─ README.md
├─ data_harvester.py
├─ install_and_launch_v475.sh
├─ loop_err.log
├─ pyproject.toml
├─ pytest.ini
└─ requirements.txt


---

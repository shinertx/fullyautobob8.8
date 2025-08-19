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

## 🔄 Recent Enhancements (still v4.7.5)

* Dynamic, self-healing venue symbol mapping (falls back to static YAML) reduces symbol rot.
* Harvester: Removed insecure eval() for EIL ingestion (safe JSON / literal parser).
* Harvester: Timestamp coercion + non-finite filtering before QA gate.
* LLM proposer: HTTP error surfacing (raise_for_status) + telemetry counters.
* Dashboard banner clarifies dynamic mapping enabled (still paper mode, no live trading).

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

---

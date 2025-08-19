---
applyTo: "**"
description: "v26meme 4.7.5 — Alpha-factory coding doctrine for Copilot"
---

# Prime Directive
Maximize **compounding per unit risk** by discovering, validating, and allocating to **causal, PIT‑correct** edges with strict anti‑ruin controls. Scoreboard: **$200 → $1M in 30 days**. Paper-first; live flip explicit.

# Non‑Negotiables
- **PIT correctness:** No forward leakage. All features/labels must be lagged/embargoed. Add tests for PIT.
- **No magic numbers:** Replace constants with **adaptive knobs** in config (with min/max), or derive from data.
- **Reproducibility:** Fixed seeds, pinned deps, deterministic code paths. No hidden global state.
- **Risk rails:** Enforce kill switches, equity floors, daily stops, lane budgets, max order notional caps.
- **Promotion discipline:** Panel CV + embargo, BH‑FDR control, robustness prober, promotion‑debt guard.
- **Orthogonality:** Prefer new alphas that reduce portfolio factor concentration (factor‑aware penalties).
- **Event‑sourced data:** Harvester is adaptive, resumable, canonical, with QA gates and retention.
- **Paper-first:** Never enable live trading, leverage, or destructive ops unless explicitly asked.

# Delivery Style
- Prefer **precise unified diffs** ( `--- a/...` / `+++ b/...` ) over full files unless new.
- Ship **runnable** code + **unit tests**. No TODOs/placeholders.
- Keep **interfaces stable**; note migrations when required.
- Emit **checklists** for ops and measurable success gates.

# Review Checklist (apply to every change)
- PIT tests pass (no leakage).  
- No new magic numbers (adaptive knob or config added).  
- FDR gates + prober thresholds unchanged or justified.  
- Risk rails enforced at portfolio + per‑alpha + lane levels.  
- LLM path hardened: JSON parse, token/latency caps, telemetry.  
- Harvester schemas validated; checkpoints & retention upheld.  
- Factor concentration delta ≤ allowed budget.

# Language/Stack Preferences
- Python 3.10+, typed where practical, Pydantic for configs.  
- Minimal dependencies; all pinned in `requirements.txt`.  
- Streamlit dashboard; Redis for state; CCXT for venues.  
- Tests via `pytest`, fast and deterministic.

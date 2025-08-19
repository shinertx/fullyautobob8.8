---
mode: ask
description: "Repo kickoff: verify wiring and mission alignment"
---
**Task:** Validate v26meme 4.7.5 repo wiring and mission alignment.

1) Confirm configs parse, deps pinned, and Redis reachable.
2) Smoke-harvest one symbol across lane TFs and write parquet + _quality.json.
3) Build features for BTC_USD_SPOT 1h with PIT tests.
4) Run one loop cycle in paper mode; ensure no live paths used.
5) Report: promoted alphas (count), FDR threshold, prober stats, factor concentration, lane budgets.

Deliver: the exact shell commands & expected outputs, then any diffs needed to fix failures.

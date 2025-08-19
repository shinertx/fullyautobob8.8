---
applyTo: "**"
description: "Engineering guardrails & diff etiquette"
---
- Generate **unified patches**; include any new imports/exports across files.
- Maintain **strict typing** on public functions; add docstrings with PIT notes.
- **Never** introduce time‑dependent branching that breaks deterministic tests.
- When touching configs: update `README.md` and add a `migration-notes.md` entry.

---
mode: ask
description: "Harvester refactor request"
---
Refactor harvester per doctrine:
- Event-sourced queue, lane TFs, resumable checkpoints
- QA gates → fail closed
- Canonical joins only
- Retention + compaction
- Token bucket
- Tests under tests/data/

Return precise diffs + tests + README updates.

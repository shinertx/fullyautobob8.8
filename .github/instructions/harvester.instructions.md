---
applyTo: "v26meme/data/**,data_harvester.py,configs/**"
description: "Event-sourced harvester doctrine"
---
- Event-sourced queue: screener snapshots + listings feed → (exchange,symbol,tf) tasks.
- Lane TFs: core [1m,5m,15m,1h,4h,1d], moonshot [1m,5m,15m], EIL on-demand.
- Resumable per (exchange,symbol,tf) checkpoints (Redis key or parquet-side metadata).
- QA gates: UTC monotonic index, schema check, gap detection; **fail closed** (no writes) if bad.
- Canonical joins via central registry/factory only.
- Storage: parquet/snappy, partitioned, retention for 1m/5m; compaction nightly.
- Token-bucket rate limiting per exchange; respect `rateLimit`.
- Tests: checkpoint resume, schema validation, canonical mapping, retention pruning.

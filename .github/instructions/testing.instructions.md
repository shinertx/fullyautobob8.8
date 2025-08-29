---
applyTo: "tests/**,v26meme/**"
description: "Test requirements"
---
- Add PIT safety tests for every feature family using synthetic time series.
- Add unit tests for promotion gates (BH-FDR) with controlled p‑value sets.
- Add simulator tests that calibrate slippage from micro‑live tables.
- Tests must run <60s on a laptop; mark slow tests and gate in CI nightly.
- Do not create new test files unless needed; extend existing suites where appropriate. Verify an existing test module doesn’t already cover the scope before adding a new file.

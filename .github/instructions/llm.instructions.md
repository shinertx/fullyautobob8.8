---
applyTo: "v26meme/llm/**,v26meme/research/**,v26meme/cli.py"
description: "LLM usage hardening & telemetry"
---
- Provider **OpenAI only**; read API key from `.env`.
- Force **hard JSON** schema; retry with temperature=0 if parse fails.
- Cap suggestions per cycle; log tokens, latency, acceptance rate.
- Disallow prompt chains that include live secrets or PII.
- Add unit tests that simulate malformed JSON from LLM and ensure graceful fallback to local heuristics.

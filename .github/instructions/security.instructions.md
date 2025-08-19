---
applyTo: "**"
description: "Security & secrets"
---
- No keys in code, tests, or fixtures. Use `.env` or GCP Secret Manager hook only.
- Add secret scanning workflow; block merges if keys detected.
- Sanitize logs: never log credentials, request bodies with secrets, or raw LLM prompts containing sensitive data.

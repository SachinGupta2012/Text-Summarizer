# Summarizer API

Phase 1 exposes only liveness and readiness probes. Database access, queues,
authentication, and summarization endpoints are intentionally deferred to later
phases described in the root roadmap.

Run locally after installing the workspace:

```powershell
uv run uvicorn summarizer_api.main:app --app-dir services/api/src --reload
```

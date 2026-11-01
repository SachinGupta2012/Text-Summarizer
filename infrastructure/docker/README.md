# Docker development environment

Phase 1 runs only `api` and `web`. PostgreSQL, Redis, and the worker are added in
Phase 7 after the offline summarization pipeline is validated.

No persistent volume is created by the Phase 1 Compose project. Docker Desktop's
host disk location is a workstation concern and is intentionally absent from the
portable Compose file.

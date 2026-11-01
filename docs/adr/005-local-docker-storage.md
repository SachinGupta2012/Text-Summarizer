# ADR 005: Use local Docker storage on D: during development

- Status: Accepted

## Decision

Run application dependencies in Docker Desktop and store its WSL virtual disk on
the D: drive. Use Docker named volumes for Linux data services rather than direct
Windows bind mounts.

## Rationale

The D: drive has more capacity, and named volumes avoid PostgreSQL permission and
performance problems on Windows-mounted filesystems. Application configuration
remains portable and does not hard-code the host path.

# ADR 001: Use a monorepo with explicit component boundaries

- Status: Accepted

## Decision

Keep the Next.js application, FastAPI service, worker boundary, ML lifecycle code,
contracts, infrastructure, and documentation in one repository with independent
packages.

## Rationale

The learning project benefits from atomic contract changes and one review history,
while package boundaries prevent web, database, and training concerns from being
coupled. A multi-repository deployment model would add coordination cost without
improving this project's scale.

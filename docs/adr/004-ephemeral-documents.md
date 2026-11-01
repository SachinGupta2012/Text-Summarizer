# ADR 004: Process source documents ephemerally

- Status: Accepted

## Decision

Do not retain original uploads or extracted raw text. Persist only metadata,
structured summaries, provenance, and user feedback.

## Rationale

Ephemeral processing reduces laptop storage, privacy exposure, backup size, and
retention complexity. Future document chat will use bounded session retention or
require re-upload rather than silently changing this policy.

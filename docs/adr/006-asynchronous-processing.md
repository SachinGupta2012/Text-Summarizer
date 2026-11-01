# ADR 006: Run document summarization asynchronously

- Status: Accepted

## Decision

The API will validate and enqueue document work. A separate worker will run
extraction, chunking, inference, consolidation, and cleanup.

## Rationale

Document processing and model inference are too slow and failure-prone for an HTTP
request thread. Asynchronous jobs allow progress states, bounded retries,
timeouts, cancellation, idempotency, and independent worker resource limits.
Implementation begins only after the offline summarization pipeline is validated.

# Summarization data flow

This document describes the target flow implemented incrementally in Phases 6
through 9.

```mermaid
sequenceDiagram
    participant U as User
    participant W as Web
    participant A as API
    participant Q as Queue
    participant K as Worker
    participant P as Document pipeline
    participant D as PostgreSQL

    U->>W: Submit text or document
    W->>A: Validated multipart request
    A->>A: Stream to UUID temporary path
    A->>D: Save metadata and pending job
    A->>Q: Enqueue job identifier
    A-->>W: Return accepted job
    Q->>K: Deliver job
    K->>P: Extract, chunk, summarize, consolidate
    P-->>K: Structured summary with citations
    K->>D: Save result and model provenance
    K->>K: Delete temporary content
    W->>A: Poll or subscribe to job state
    A-->>W: Return result
```

The API rejects unsupported media and oversized input before queueing. The worker
cleans up on success, failure, timeout, and cancellation. A startup sweeper will
remove abandoned temporary content after a bounded TTL. Raw document content is
excluded from logs, metrics, audit events, and database backups.

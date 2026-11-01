# Security boundaries

## Trust model

Uploaded documents, filenames, URLs, extracted text, and model output are
untrusted. They cannot supply filesystem paths, commands, configuration, or
authorization decisions.

## Target controls

- The browser communicates only with versioned API routes.
- Authentication and ownership checks occur in FastAPI.
- Passwords use Argon2id; refresh tokens are rotated and stored only as hashes.
- Uploads are streamed to generated temporary names with size and type limits.
- The worker receives job identifiers rather than raw content in queue messages.
- Database queries use parameterized repositories.
- Secrets are injected at runtime and never committed.
- Logs, traces, metrics, and audit events exclude raw documents and credentials.
- Containers run without unnecessary privileges and expose minimal ports.

Phase 1 contains no credentials, authentication, uploads, database, or public
deployment. Its security work is limited to explicit boundaries, secret-safe
configuration, dependency isolation, and non-root container design.

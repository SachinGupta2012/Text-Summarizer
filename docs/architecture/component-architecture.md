# Component architecture

## Web application

The Next.js application will eventually provide authentication, uploads, job
progress, summary results, history, and feedback. It never accesses PostgreSQL,
Redis, temporary files, or model artifacts directly.

## API service

FastAPI is the public application boundary. Later phases add request validation,
authorization, metadata persistence, idempotent job creation, and result reads.
The API does not perform long-running extraction or inference in request threads.

## Worker

The worker will claim jobs, resolve a temporary source, run the validated document
pipeline, persist structured results, and delete transient content in a `finally`
path. Messages contain identifiers and safe internal references, not document
bodies.

## ML package

The ML package is usable from Kaggle, command-line evaluation, and the worker. It
must not depend on web frameworks or database repositories. Kaggle notebooks call
package entry points rather than defining a second implementation.

## Data services

PostgreSQL will store users, sessions, document metadata, jobs, summaries, model
versions, feedback, and audit events. Redis will provide queue state, rate limits,
and short-lived coordination. Neither service is introduced until the feature
requirements exist.

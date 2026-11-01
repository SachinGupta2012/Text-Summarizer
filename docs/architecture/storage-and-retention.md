# Storage and retention

## Local development

Docker Desktop stores its virtual disk on the D: drive. Compose uses named volumes
when PostgreSQL and Redis arrive in Phase 7; no Windows bind mount is used for the
PostgreSQL data directory.

## Retention classes

| Data                   | Target retention                                                     |
| ---------------------- | -------------------------------------------------------------------- |
| Original upload        | Processing lifetime only                                             |
| Extracted raw text     | Processing lifetime only                                             |
| Temporary chunks       | Processing lifetime plus bounded cleanup TTL                         |
| Document metadata      | Until the learning user deletes it                                   |
| Summary and provenance | Until the learning user deletes it                                   |
| Training datasets      | Versioned outside Git; retained only as required for reproducibility |
| Model checkpoints      | Temporary Kaggle artifacts; promoted adapters retained separately    |
| Application logs       | Rotated and content-free                                             |

Phase 1 creates no persistent volume. Later PostgreSQL development data is
reproducible from migrations and seed scripts. A compressed logical backup is
created only before risky migrations, with at most two local copies. Local backup
on D: protects against a bad migration, not drive failure.

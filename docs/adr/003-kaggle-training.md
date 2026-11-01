# ADR 003: Use Kaggle GPU for evaluation and QLoRA training

- Status: Accepted

## Decision

Run base-model evaluation and QLoRA training on Kaggle T4 GPU sessions. Keep
training code in the repository and use notebooks only as launchers.

## Rationale

The development laptop is CPU-only, and the standard QLoRA stack is CUDA-centric.
Kaggle provides suitable temporary GPU access. Checkpointing and resumability are
required because sessions and weekly quotas are bounded.

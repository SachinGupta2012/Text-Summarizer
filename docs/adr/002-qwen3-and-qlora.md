# ADR 002: Use Qwen3 4B Instruct with QLoRA

- Status: Accepted

## Decision

Use `Qwen/Qwen3-4B-Instruct-2507` as the primary base model and demonstrate
fine-tuning through a QLoRA adapter rather than full-parameter training.

## Rationale

The decoder-only Transformer supports instruction-style multi-domain tasks and
has a permissive model license. QLoRA makes a measurable fine-tuning experiment
feasible on temporary T4 GPUs while preserving an untouched baseline for a fair
comparison. Selection does not guarantee improvement; Phase 5 has an explicit
promotion gate.

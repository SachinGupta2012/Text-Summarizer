# Product scope

## Purpose

This repository is a learning and portfolio project. It should demonstrate a
disciplined data-science lifecycle and production-style software boundaries
without pretending to operate commercial infrastructure.

## Target summarization tasks

The eventual system should summarize:

- general articles and reports;
- text extracted from PDF and DOCX files;
- long and structured documents;
- legal-style documents;
- customer-support conversations;
- multi-speaker meetings; and
- documents containing domain-specific terminology.

Document-grounded chat is a separate feature after summarization is complete and
validated.

## Quality goals

The project must show whether fine-tuning improves the selected base model. A
claim of improvement requires a fixed evaluation set, identical inference
settings, automatic metrics, factuality checks, per-domain analysis, human
review, and performance measurements.

## Constraints

- Development happens on a CPU laptop with limited local storage.
- GPU evaluation and QLoRA training run on Kaggle T4 GPUs.
- Original user documents are processed ephemerally and are not retained.
- The application is local during development; paid cloud infrastructure is not
  required.
- Data and model licenses are recorded even though the project is non-commercial.

## Non-goals

- Training a foundation model from scratch
- Full-parameter fine-tuning of Qwen3 4B
- Claiming legal or professional advice from summaries
- Retaining sensitive source documents
- Building RAG before summarization quality is established
- Kubernetes or high-availability infrastructure for the learning environment

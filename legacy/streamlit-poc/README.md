# Legacy Streamlit proof of concept

This directory preserves the project state tagged as `legacy-streamlit-poc`.

The original application loaded `philschmid/bart-large-cnn-samsum` from the
Streamlit request path and generated summaries. Although files named `trainer.py`
and `train_pipeline.py` existed, the executed pipeline sampled five SAMSum rows
and performed inference; it did not tokenize a training dataset, invoke the
trainer, evaluate a checkpoint, or save a fine-tuned model.

The snapshot remains available for comparison and historical context. It is not
part of the new runtime, dependency graph, Docker build, or CI checks. See
`ORIGINAL_README.md` for the documentation that accompanied the POC.

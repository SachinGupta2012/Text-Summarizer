# ML package

This package will own dataset preparation, evaluation, QLoRA training, and
inference logic in later phases. Phase 1 deliberately contains only stable project
identifiers and package-import tests. It does not install PyTorch, Transformers,
PEFT, TRL, or bitsandbytes and does not download any model or dataset.

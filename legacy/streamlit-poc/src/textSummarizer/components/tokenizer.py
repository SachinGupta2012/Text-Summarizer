def preprocess_batch(batch, tokenizer):
    inputs = tokenizer(
        ["summarize: " + text for text in batch["dialogue"]],
        truncation=True,
        padding="max_length",
        max_length=512
    )

    targets = tokenizer(
        batch["summary"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

    inputs["labels"] = targets["input_ids"]
    return inputs

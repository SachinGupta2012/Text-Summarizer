import torch

def summarize(text, model, tokenizer):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Tokenize and move inputs to device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate summary
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=40,
        num_beams=4,
        repetition_penalty=2.0,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

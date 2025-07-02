from transformers import Trainer, TrainingArguments

def train_model(model, tokenizer, tokenized_dataset, config):
    args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        save_strategy="no"
    )
    trainer = Trainer(model=model, args=args, train_dataset=tokenized_dataset, tokenizer=tokenizer)
    trainer.train()
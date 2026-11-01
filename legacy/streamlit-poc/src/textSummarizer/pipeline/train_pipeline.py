import yaml
from datasets import Dataset
from src.textSummarizer.components.data_loader import load_samsum_data
from src.textSummarizer.components.model_init import load_model
from src.textSummarizer.components.tokenizer import preprocess_batch
from src.textSummarizer.components.infer import summarize


def run_pipeline():
    config = yaml.safe_load(open("config/config.yaml"))

    # Load a small dataset sample for testing
    data = load_samsum_data(sample_size=5)
    
    # Load pretrained model and tokenizer
    tokenizer, model = load_model(config['model_name'])
    
    # Just show summaries using pretrained model (no training)
    for example in data:
        text = example["dialogue"]
        print("\n📥 Input:\n", text)
        print("📤 Summary:\n", summarize(text, model, tokenizer))

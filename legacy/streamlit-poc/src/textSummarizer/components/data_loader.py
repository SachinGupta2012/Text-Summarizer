import pandas as pd
from datasets import Dataset

def load_samsum_data(sample_size=None):
    df = pd.read_csv("data/samsum-train.csv")

    # Drop 'id' and ensure proper column names
    df = df[["dialogue", "summary"]]

    if sample_size:
        df = df.head(sample_size)

    return Dataset.from_pandas(df)

# src/textSummarizer/utils/common.py

import logging
import os

def get_logger(name, log_file="logs.txt"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    return logging.getLogger(name)

import torch
import random
from transformers import (
    pipeline,
    AutoTokenizer
)

HF_PIPELINES = {}
HF_TOKENIZERS = {}

def get_hf_pipeline(model_name):
    if model_name not in HF_PIPELINES:
        hf_pipeline = pipeline("text-generation",
                               model=model_name,
                               model_kwargs={"torch_dtype": torch.bfloat16},
                               device="cuda",)
        HF_PIPELINES[model_name] = hf_pipeline
    return HF_PIPELINES[model_name]


def get_hf_tokenizer(model_name):
    if model_name not in HF_TOKENIZERS:
        hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        hf_tokenizer.pad_token = hf_tokenizer.eos_token
        HF_TOKENIZERS[model_name] = hf_tokenizer
    return HF_TOKENIZERS[model_name]


def get_synthid_config(custom_seed, ngram_len):
    rng = random.Random(custom_seed)
    keys = [int(rng.uniform(0, 1000)) for _ in range(30)]
    return {
        # This corresponds to H=4 context window size in the paper.
        "ngram_len": ngram_len,
        "keys": keys,
        "sampling_table_size": 2**16,
        "sampling_table_seed": custom_seed,
        "context_history_size": 1024,
    }
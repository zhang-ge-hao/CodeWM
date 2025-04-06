import os, sys, io
import math
from transformers import (
    SynthIDTextWatermarkLogitsProcessor,
    PreTrainedTokenizerBase
)
import torch
sys.path.append("src")
from _data_structure import *
from _util import hash_str_to_int
from _hf_obj import (
    get_hf_pipeline, 
    get_hf_tokenizer,
    get_synthid_config
)
from _sweet import (
    SweetDetector,
    WatermarkDetector
)


def synthid_compute_z_score(observed_g_tensor: torch.Tensor, 
                            coinflip_prob: float = 0.5) -> float:
    # for Bernoulli distribution
    observed_g_tensor = observed_g_tensor.reshape(-1)
    x = observed_g_tensor.float().sum().item()
    n = observed_g_tensor.size(-1)
    p = coinflip_prob
    exp = n * p
    std = math.sqrt(n * p * (1 - p))
    z_score = (x - exp) / std
    return z_score


def tokenize(example: str, tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    inputs = tokenizer(
        example,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    return inputs["input_ids"].squeeze()


def calculate_entropy(model, tokenized_text):
    with torch.no_grad():
        output = model(torch.unsqueeze(tokenized_text, 0), return_dict=True)
        probs = torch.softmax(output.logits, dim=-1)
        entropy = -torch.where(probs > 0, probs *
                               probs.log(), probs.new([0.0])).sum(dim=-1)
        return entropy[0].cpu().tolist()


def detect(task: GenTask|ObfTask, gen_task: GenTask, z_threshold=4.0, ngram_len=5):
    assert all(o is not None for o in [
        task.p4d, task.g4d, task.solution])
    if not gen_task.need_obf or gen_task.watermarking == "no_wm":
        # NOTE need_obf == need_detect in current setting
        return
    tokenizer = get_hf_tokenizer(task.model_name)
    custom_seed = hash_str_to_int(task.id)
    if gen_task.watermarking == "synthid":
        output_text = task.g4d
        output_ids = tokenizer(output_text, return_tensors="pt").input_ids
        output_ids = output_ids.to("cuda")

        if output_ids.size(-1) < ngram_len:
            task.res_d = 0
        else:
            config_dict = get_synthid_config(custom_seed, ngram_len)
            synthid_processor = SynthIDTextWatermarkLogitsProcessor(device="cuda",
                                                                    **config_dict)

            g_tensor = synthid_processor.compute_g_values(output_ids) 

            z_score = synthid_compute_z_score(g_tensor)
            task.res_d = z_score
    elif gen_task.watermarking in ["sweet", "wllm"]:
        
        vocab = list(tokenizer.get_vocab().values())
        tokenized_prefix = tokenize(task.p4d, tokenizer)
        tokenized_suffix = tokenize(task.g4d, tokenizer)
        if len(task.g4d) == 0 or len(tokenized_suffix.size()) == 0 or \
                tokenized_suffix.size(-1) == 0:
            tokenized_text = tokenized_prefix
        else:
            tokenized_text = torch.cat((tokenized_prefix, tokenized_suffix), dim=0)

        if gen_task.watermarking == "sweet":
            detector = SweetDetector(
                vocab=vocab,
                gamma=task.gamma,
                tokenizer=tokenizer,
                z_threshold=z_threshold,
                entropy_threshold=task.entropy_threshold,
                ngram_len=ngram_len,
                hash_key=custom_seed)
            hf_pipeline = get_hf_pipeline(task.model_name)
            entropy = calculate_entropy(hf_pipeline.model,
                                        tokenized_text.to("cuda"))
            detection_result_dict = detector.detect(tokenized_text=tokenized_text,
                                                    tokenized_prefix=tokenized_prefix,
                                                    entropy=entropy,)
        else: # wllm
            detector = WatermarkDetector(
                vocab=vocab,
                gamma=task.gamma,
                tokenizer=tokenizer,
                z_threshold=z_threshold,
                ngram_len=ngram_len,
                hash_key=custom_seed)
            detection_result_dict = detector.detect(tokenized_text=tokenized_text,
                                                    tokenized_prefix=tokenized_prefix)
        if "z_score" not in detection_result_dict:
            assert detection_result_dict["invalid"]
            task.res_d = 0
        else:
            task.res_d = detection_result_dict["z_score"]
    else:
        raise NotImplementedError()
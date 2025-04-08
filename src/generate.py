import torch
from transformers import (
    pipeline,
    StoppingCriteria,
    SynthIDTextWatermarkingConfig,
    StoppingCriteriaList,
    LogitsProcessorList
)
import logging
import sys
import json
from dataclasses import asdict
sys.path.append("src")

from _data import *
from _sweet import (
    SweetLogitsProcessor,
    WatermarkLogitsProcessor,
)
from _util import (
    dataclass_2_str
)
from _hf_obj import (
    get_hf_pipeline, 
    get_hf_tokenizer,
    get_synthid_config
)
from evaluation import evaluate
from detection import detect


def make_raw_chat_prompt(task: GenTask, tokenizer) -> str:
    task_prompt = task.ori_prompt.strip()
    language_full_name_map = {"js": "JavaScript", "py": "Python"}
    language = language_full_name_map[task.language]

    instruction_prefix = f"Please provide a self-contained {language} " + \
        "script that solves the following problem in a markdown code block:"
    response_prefix = f"Below is a {language} script with a self-contained " + \
        "function that solves the problem and passes corresponding tests:"

    _MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"

    task_prompt = f"""\
{instruction_prefix}
```
{task_prompt.strip()}
```
"""
    response = f"""\
{response_prefix}
```{{{language}}}
{_MAGIC_SPLITTER_}
```
"""
    task_prompt = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": task_prompt},
            {"role": "assistant", "content": response},
        ],
        tokenize=False,
    ).split(_MAGIC_SPLITTER_)[0]
    return task_prompt


class CodeStoppingCriteria(StoppingCriteria):
    def __init__(self, task: GenTask, tokenizer):
        super().__init__()
        assert task.p4d is not None
        self.task = task
        self.tokenizer = tokenizer
        self.step = 10
        self.generated_token_count = 0

    def __call__(self,
                 input_ids: torch.LongTensor,
                 scores: torch.FloatTensor,
                 **kwargs) -> bool:
        assert len(input_ids) == 1, "Not for batch."
        self.generated_token_count += 1
        if self.generated_token_count % self.step == 0:
            decoded_text: str = self.tokenizer.decode(
                input_ids[0], skip_special_tokens=True)
            generated_text = decoded_text[len(self.task.p4d):]
            if self.task.is_inst:
                if "\n```" in generated_text:
                    return True
            elif self.task.language == "js":
                bracket_debt = 1
                for char in generated_text:
                    if char == "{":
                        bracket_debt += 1
                    elif char == "}":
                        bracket_debt -= 1
                    if bracket_debt == 0:
                        return True
            elif self.task.language == "py":
                lines = generated_text.split('\n')
                empty_count = 0
                for i in range(len(lines) - 1, -1, -1):
                    line = lines[i]
                    if len(line.strip()) > 0:
                        empty_count = 0
                    else:
                        empty_count += 1
                    if empty_count > 4:
                        return True
                    if not len(line.strip()) == 0 and \
                            not line.startswith(" ") and \
                            not line.startswith("\t") and \
                            not line.startswith("#"):
                        return True
            else:
                raise NotImplementedError()
        return False


def get_solution(task: GenTask):
    assert task.p4d is not None
    assert task.g4d is not None
    prompt = task.p4d
    generation = task.g4d.rstrip()

    if task.is_inst:
        if "\n```" in generation:
            solution = generation.split("\n```")[0]
        elif "```" in generation:
            solution = generation.split("```")[0]
        else:
            solution = generation
        if task.language == "py":
            met_entry_point = False
            for idx in range(len(solution) - 1):
                if solution[idx: idx + len(task.entry_point)] == task.entry_point:
                    met_entry_point = True
                elif met_entry_point and solution[idx] == "\n":
                    if solution[idx + 1] not in ("\t", "#", "\n", " "):
                        return solution[: idx].rstrip()
        elif task.language == "js":
            met_entry_point = False
            met_lb = False
            bracket_debt = 1
            for idx in range(len(solution)):
                if solution[idx: idx + len(task.entry_point)] == task.entry_point:
                    met_entry_point = True
                elif not met_lb and met_entry_point and solution[idx] == "{":
                    met_lb = True
                elif met_entry_point and met_lb:
                    if solution[idx] == "{":
                        bracket_debt += 1
                    elif solution[idx] == "}":
                        bracket_debt -= 1
                    if bracket_debt == 0:
                        return solution[: idx + 1].rstrip()
        return solution
    elif task.language == "py":
        for idx in range(len(generation) - 1):
            if generation[idx] == "\n":
                if generation[idx + 1] not in ("\t", "#", "\n", " "):
                    return prompt + generation[: idx].rstrip()
        return prompt + generation
    elif task.language == "js":
        bracket_debt = 1
        for idx in range(len(generation)):
            if generation[idx] == "{":
                bracket_debt += 1
            elif generation[idx] == "}":
                bracket_debt -= 1
            if bracket_debt == 0:
                return prompt + generation[: idx + 1].rstrip()
        return prompt + generation
    else:
        raise NotImplementedError()


def generate(task: GenTask, max_new_tokens=512, ngram_len=5):
    logging.info(f"[start] {dataclass_2_str(task)}")

    custom_seed = task.custom_seed
    hf_pipeline = get_hf_pipeline(task.model_name)
    tokenizer = get_hf_tokenizer(task.model_name)

    if not task.is_inst:
        task.p4d = task.ori_prompt.strip() + "\n"
    else:
        task.p4d = make_raw_chat_prompt(task, tokenizer)
    
    stopping_criteria_list = StoppingCriteriaList(
        [CodeStoppingCriteria(task, tokenizer)])

    assert task.temperature is not None
    if task.watermarking == "synthid":
        config_dict = get_synthid_config(custom_seed, ngram_len)
        wm_config = SynthIDTextWatermarkingConfig(**config_dict)
        wm_param = {
            "watermarking_config": wm_config
        }
    elif task.watermarking == "sweet":
        assert all(p is not None for p in [
            task.gamma, task.delta, task.entropy_threshold])
        vocab = list(tokenizer.get_vocab().values())
        logits_processor = SweetLogitsProcessor(
            vocab=vocab,
            gamma=task.gamma,
            delta=task.delta,
            entropy_threshold=task.entropy_threshold,
            ngram_len=ngram_len,
            hash_key=custom_seed)
        wm_param = {
            "logits_processor": LogitsProcessorList([logits_processor])
        }
    elif task.watermarking == "wllm":
        assert all(p is not None for p in [task.gamma, task.delta])
        vocab = list(tokenizer.get_vocab().values())
        logits_processor = WatermarkLogitsProcessor(
            vocab=vocab,
            gamma=task.gamma,
            delta=task.delta,
            ngram_len=ngram_len,
            hash_key=custom_seed)
        wm_param = {
            "logits_processor": LogitsProcessorList([logits_processor])
        }
    elif task.watermarking == "no_wm":
        wm_param = {}
    else:
        raise NotImplementedError()
    
    assert task.temperature is not None
    task.g4d = hf_pipeline(task.p4d,
                           do_sample=True, max_new_tokens=max_new_tokens,
                           temperature=task.temperature,
                           pad_token_id=tokenizer.eos_token_id,
                           stopping_criteria=stopping_criteria_list,
                           **wm_param)[0]["generated_text"][len(task.p4d): ]
    task.solution = get_solution(task)
    task.s_len = len(tokenizer.encode(task.solution))
    logging.info(f"[generated] {dataclass_2_str(task)}")

    evaluate(task=task, gen_task=task)
    logging.info(f"[evaluated] {dataclass_2_str(task)}")

    if task.need_obf and task.watermarking != "no_wm":
        # NOTE need_obf == need_detect in current setting
        detect(task=task, gen_task=task)
        logging.info(f"[detected] {dataclass_2_str(task)}")

import sys
import json
import re
import os
from dataclasses import asdict
from itertools import product
import random
sys.path.append("src")

from _dataclass import *

def extract_entry_point(dp, ds_name):
    if "entry_point" in dp:
        return dp["entry_point"]
    assert ds_name in ["humaneval_py", "humaneval_js", "mbpp_js"]
    prompt = dp["prompt"]
    if ds_name == "mbpp_js": # MultiPL-E
        name = dp["name"]
        return "_".join(name.split("_")[2: ])
    elif ds_name == "humaneval_py":
        prompt_tokens = re.split(r"\s+|\(", prompt)
        def_idx = None
        for idx, token in enumerate(prompt_tokens):
            if token == "def":
                def_idx = idx
        if def_idx is not None and def_idx < len(prompt_tokens):
            return prompt_tokens[def_idx + 1]
    elif ds_name == "humaneval_js":
        declaration = prompt.strip().split("\n")[-1]
        declaration_tokens = re.split(r"\s+", declaration)
        const_idx = [idx for idx, token in enumerate(declaration_tokens)
                    if token == "const"][0]
        return declaration_tokens[const_idx + 1]
    raise RuntimeError(f"parse_function_name failed on {prompt}.")


def extract_prompt(dp, ds_name):
    if ds_name == "mbpp_py": # mbpp-plus
        entry_point = dp["entry_point"]
        prompt_lines = []
        for line in dp["canonical_solution"].split("\n"):
            prompt_lines.append(line)
            if entry_point in line:
                break
        for line in dp["prompt"].split("\n"):
            prompt_lines.append("    " + line.strip())
        return "\n".join(prompt_lines)
    return dp["prompt"]


def extract_num_id(dp, ds_name):
    if ds_name == "mbpp_js": # MultiPL-E
        return dp["name"].split("_")[1]
    return dp["task_id"].split("/")[1]


def extract_test(dp, ds_name):
    if ds_name == "mbpp_js": # MultiPL-E
        return dp["tests"]
    elif ds_name == "mbpp_py": # mpbb-plus
        return dp["assertion"]
    return dp["test"]


if __name__ == "__main__":
    models = [ # and is_inst
        ("meta-llama/Llama-3.1-8B-Instruct", True),
        ("deepseek-ai/deepseek-coder-33b-base", False)
    ]

    datasets = [ # and language
        ("humaneval-x_py.jsonl", "humaneval_py", "py"), 
        ("humaneval-x_js.jsonl", "humaneval_js", "js"), 
        ("mbppp_py.jsonl", "mbpp_py", "py"), 
        ("mbpp-MuE_js.jsonl", "mbpp_js", "js"), 
    ]

    wms = [ # and need_obf
        ("no_wm", False),
        ("synthid", True), 
        ("wllm", True), 
        ("sweet", True)
    ]

    lang_2_obf = {
        "py": ["pyminify", "pyminifier"],
        "js": ["javascript-obfuscator", "uglifyjs"]
    }

    para_candidate = {
        "delta": [0.5, 1.0, 2.0, 3.0, 4.0],
        "gamma": [0.1, 0.25, 0.5],
        "entropy_threshold": [0.3, 0.6, 0.9, 1.2],
        "temperature": [0.25, 0.5, 0.75, 1.0, 1.25],
        # "max_new_tokens": [512],
        # "z_threshold": [4],
        # "ngram_len": [5]
    }

    model_2_shorter_name = {
        "meta-llama/Llama-3.1-8B-Instruct": "Llama31Instruct8B",
        "deepseek-ai/deepseek-coder-33b-base": "DSCoderBase33B",
    }

    ds_name_2_data = {}
    for file_name, ds_name, lang in datasets:
        ds_name_2_data[ds_name] = []
        with open(f"data/original/{file_name}") as ds_file:
            for line in ds_file:
                dp = json.loads(line)

                num_task_id = extract_num_id(dp, ds_name)
                ori_prompt = extract_prompt(dp, ds_name)
                entry_point = extract_entry_point(dp, ds_name)
                test = extract_test(dp, ds_name)

                ds_name_2_data[ds_name].append((
                    f"{ds_name}/{num_task_id}", 
                    ori_prompt, 
                    entry_point, 
                    test
                ))
    wm_name_2_para_comb = {}
    for wm_name, need_obf in wms:
        wm_name_2_para_comb[wm_name] = []
        if wm_name in ["no_wm", "synthid"]:
            for temperature in para_candidate["temperature"]:
                wm_name_2_para_comb[wm_name].append((
                    temperature, None, None, None
                ))
        elif wm_name == "wllm":
            for delta, gamma in product(para_candidate["delta"], para_candidate["gamma"]):
                wm_name_2_para_comb[wm_name].append((
                    1.0, delta, gamma, None
                ))
        elif wm_name == "sweet":
            for delta, gamma, entropy_threshold in product(para_candidate["delta"], 
                                                           para_candidate["gamma"], 
                                                           para_candidate["entropy_threshold"]):
                wm_name_2_para_comb[wm_name].append((
                    1.0, delta, gamma, entropy_threshold
                ))

    gen_task_id_set = set()
    gen_task_count, obf_task_count = 0, 0

    for model, is_inst in models:
        for file_name, ds_name, lang in datasets:
            for wm_name, need_obf in wms:
                # The prompts of MultiPL-E JS are simple.
                # The inst scheme will generate wrong function name for MultiPL-E JS.
                if ds_name == "mbpp_js": # MultiPL-E
                    is_inst = False

                para_comb_count = len(wm_name_2_para_comb[wm_name])
                ds_task_count = len(ds_name_2_data[ds_name])
                task_folder_name = f"{model_2_shorter_name[model]}--{wm_name}--{ds_name}"

                print(f"{task_folder_name}: " + \
                      f"{para_comb_count} x {ds_task_count} = {para_comb_count * ds_task_count}")

                for __dp_idx, (temperature, delta, gamma, entropy_threshold) in enumerate(wm_name_2_para_comb[wm_name]):
                    dp_idx, dp_gen_tasks, dp_obf_tasks = __dp_idx + 1, [], []
                    for task_name, ori_prompt, entry_point, test in ds_name_2_data[ds_name]:

                        gen_task_id = f"{task_name}--{model}--{wm_name}--" + \
                            f"{temperature}/{delta}/{gamma}/{entropy_threshold}"
                        
                        assert gen_task_id not in gen_task_id_set, gen_task_id
                        gen_task_id_set.add(gen_task_id)

                        gen_task = GenTask(
                            id=gen_task_id,
                            task_name=task_name, dataset_name=ds_name, 
                            model_name=model, watermarking=wm_name, 
                            language=lang, is_inst=is_inst, need_obf=need_obf,
                            temperature=temperature, delta=delta, 
                            gamma=gamma, entropy_threshold=entropy_threshold,
                            ori_prompt=ori_prompt, entry_point=entry_point, test=test,
                            p4d=None, g4d=None, solution=None, s_len=None,
                            passed=None, z_score=None, p_value=None,
                            custom_seed=int(random.uniform(0, 0xFFFFFFFFFFFFFFFF))
                        )
                        dp_gen_tasks.append(gen_task)

                        if not need_obf:
                            continue
                        for obf_name in lang_2_obf[lang]:
                            obf_task_id = f"{obf_name}--{gen_task_id}"
                            obf_task = ObfTask(
                                id=obf_task_id, gen_task_id=gen_task_id, obf_name=obf_name,
                                p4d=None, g4d=None, solution=None, s_len=None,
                                passed=None, z_score=None, p_value=None, bad_trans=None
                            )
                            dp_obf_tasks.append(obf_task)
                        
                    gen_task_count += len(dp_gen_tasks)
                    obf_task_count += len(dp_obf_tasks)
                    for task_type, tasks in [("generate", dp_gen_tasks), ("obfuscate", dp_obf_tasks)]:
                        dp_file_path = f"data/task/{task_folder_name}/{dp_idx:03d}/{task_type}.jsonl"
                        os.makedirs(os.path.dirname(dp_file_path), exist_ok=True)
                        with open(dp_file_path, "w") as file:
                            for task in tasks:
                                file.write(json.dumps(asdict(task)) + "\n")

    print(f"COUNT(gen_tasks): {gen_task_count}")
    print(f"COUNT(obf_tasks): {obf_task_count}")

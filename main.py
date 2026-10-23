import logging
import argparse
import sys, os
sys.path.append("src")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_name", type=str, required=True)
    parser.add_argument("-d", "--dp_idx", type=str, required=True)
    parser.add_argument("--obf_only", action='store_true',)
    parser.add_argument("--log_file", action='store_true',)
    parser.add_argument("--debug_mode", action='store_true',)
    return parser.parse_args()


def set_logging(log_file: bool):
    if not log_file:
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(message)s')
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join("log", f'{timestamp}.log')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_path, encoding='utf-8')
            ]
        )


ARGS = parse_args()
set_logging(ARGS.log_file)


from typing import List
import json
from tqdm import tqdm
import traceback
from dataclasses import asdict
from itertools import product

from _data import *
from generate import generate
from obfuscate import obfuscate
from metrics import calculate


if __name__ == "__main__":
    task_name = ARGS.task_name
    dp_idx = ARGS.dp_idx
    obf_only = ARGS.obf_only
    debug_mode = ARGS.debug_mode

    gen_dp_file_path = f"data/task/{task_name}/{dp_idx}/generate.jsonl"
    obf_dp_file_path = f"data/task/{task_name}/{dp_idx}/obfuscate.jsonl"

    gen_dp_result_path = f"data/result/{task_name}/{dp_idx}/generate.jsonl"
    obf_dp_result_path = f"data/result/{task_name}/{dp_idx}/obfuscate.jsonl"
    metrics_result_path = f"data/result/{task_name}/{dp_idx}/metrics.jsonl"

    if obf_only:
        gen_dp_file_path = gen_dp_result_path
    else:
        if os.path.exists(gen_dp_result_path):
            os.remove(gen_dp_result_path)
    if os.path.exists(obf_dp_result_path):
        os.remove(obf_dp_result_path)

    os.makedirs(os.path.dirname(gen_dp_result_path), exist_ok=True)
    os.makedirs(os.path.dirname(obf_dp_result_path), exist_ok=True)

    with open(gen_dp_file_path) as file:
        gen_tasks = [GenTask(**json.loads(l)) for l in file]
        id_2_gen_task = {gt.id: gt for gt in gen_tasks}
    with open(obf_dp_file_path) as file:
        obf_tasks = [ObfTask(**json.loads(l)) for l in file]
    
    if debug_mode:
        gen_tasks = gen_tasks[: 10]
        gen_task_ids = set([t.id for t in gen_tasks])
        obf_tasks = [t for t in obf_tasks if t.gen_task_id in gen_task_ids]

    if not obf_only:
        with open(gen_dp_result_path, "a") as file:
            for gen_task in tqdm(gen_tasks):
                try:
                    generate(gen_task)
                except Exception as e:
                    logging.error(traceback.format_exc())
                file.write(json.dumps(asdict(gen_task)) + "\n")
                file.flush()

    with open(obf_dp_result_path, "a") as file:
        for obf_task in tqdm(obf_tasks):
            try:
                gen_task = id_2_gen_task[obf_task.gen_task_id]
                obfuscate(obf_task, gen_task)
            except Exception as e:
                logging.error(traceback.format_exc())
            file.write(json.dumps(asdict(obf_task)) + "\n")
            file.flush()

    try:
        dps = calculate(gen_tasks, obf_tasks)
        with open(metrics_result_path, "w") as file:
            for dp in dps:
                file.write(json.dumps(asdict(dp)) + "\n")
    except Exception as e:
        logging.error(traceback.format_exc())

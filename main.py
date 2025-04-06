from typing import List
import argparse
import sys
import os
import json
from tqdm import tqdm
import traceback
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
from dataclasses import asdict
from itertools import product
sys.path.append("src")

from _data_structure import *
from generate import generate
from obfuscate import obfuscate

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_name", type=str, required=True)
    parser.add_argument("-d", "--dp_idx", type=str, required=True)
    args = parser.parse_args()

    gen_dp_file_path = f"data/task/{args.task_name}/{args.dp_idx}/generate.jsonl"
    obf_dp_file_path = f"data/task/{args.task_name}/{args.dp_idx}/obfuscate.jsonl"

    gen_dp_result_path = f"data/result/{args.task_name}/{args.dp_idx}/generate.jsonl"
    obf_dp_result_path = f"data/result/{args.task_name}/{args.dp_idx}/obfuscate.jsonl"

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

    with open(gen_dp_result_path, "a") as file:
        for gen_task in tqdm(gen_tasks):
            try:
                generate(gen_task)
                file.write(json.dumps(asdict(gen_task)) + "\n")
                file.flush()
            except Exception as e:
                logging.error(traceback.format_exc())

    with open(obf_dp_result_path, "a") as file:
        for obf_task in tqdm(obf_tasks):
            try:
                gen_task = id_2_gen_task[obf_task.gen_task_id]
                obfuscate(obf_task, gen_task)
                file.write(json.dumps(asdict(obf_task)) + "\n")
                file.flush()
            except Exception as e:
                logging.error(traceback.format_exc())

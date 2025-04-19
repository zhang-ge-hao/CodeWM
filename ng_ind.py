import os
import argparse
import json
import math
import subprocess
import shutil
import logging
import uuid
import random
import re
import sys
from typing import Dict, List, Tuple
from transformers import AutoTokenizer
from scipy.stats import chi2_contingency
from scipy import stats
import numpy as np

from openai import OpenAI
from tqdm import tqdm
import contextlib
import signal
import io
import time
from subprocess import TimeoutExpired
from dataclasses import dataclass, asdict
sys.path.append("src")
from _util import (
    create_tempdir,
    change_dir,
    dataclass_2_str
)



class WriteOnlyStringIO(io.StringIO):
    """ StringIO that throws an exception when it's read from """

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """ Returns True if the IO object can be read. """
        return False

class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = 'stdin'


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield

@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


class TimeoutException(Exception):
    pass


def hash_string_to_int(string):
    import hashlib
    hash_obj = hashlib.sha3_512(string.encode())
    hash_int = int(hash_obj.hexdigest(), 16)
    return hash_int & 0xFFFFFFFFFFFFFFFF


def hash_integer_array_to_int(arr):
    import hashlib
    arr_str = ','.join(map(str, arr))
    hash_obj = hashlib.sha3_512(arr_str.encode())
    hash_int = int(hash_obj.hexdigest(), 16)
    return hash_int & 0xFFFFFFFFFFFFFFFF


def extract_entry_point(prompt: str):
    declaration = prompt.strip().split("\n")[-1]
    declaration_tokens = re.split(r"\s+", declaration)
    const_idx = [idx for idx, token in enumerate(declaration_tokens)
                if token == "const"][0]
    return declaration_tokens[const_idx + 1]


def obfuscate(prompt: str, ori_code: str):
    entry_point = extract_entry_point(prompt)
    with create_tempdir() as work_dir:
        solution_file_name = "solution.js"
        with open(solution_file_name, "w") as file:
            file.write(ori_code)
        cmd = [
            "uglifyjs", "-c", "-b",
            "-m", f"reserved=[{entry_point}]",
            "--", solution_file_name
        ]
        result = subprocess.run(cmd, timeout=10, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Error during obfuscation:\n{result.stderr}")
        else:
            return result.stdout


def test(code_with_test: str) -> bool:
    with create_tempdir():
        code_file_name = "test.js"
        with open(code_file_name, "w") as file:
            file.write(code_with_test)
        try:
            exec_result = None
            with time_limit(10):
                exec_result = subprocess.run([f"node", code_file_name], 
                                            timeout=10, 
                                            capture_output=True)
            if exec_result.stderr.decode():
                # logging.warning(exec_result.stderr.decode())
                return False
            elif exec_result.stdout.decode():
                # logging.warning(exec_result.stdout.decode())
                return False
            else:
                return True
        except TimeoutException:
            # logging.warning("Timeout.")
            return False
        except BaseException:
            # logging.warning("Other exception.")
            return False


def compute_z_score(x, n, p):
    std = math.sqrt(n * p * (1 - p))
    exp = n * p
    z = (x - exp) / std
    return z


def detect(generated_code, n_gram):
    tokens = tokenizer.encode(generated_code)
    if len(tokens) < n_gram:
        return None, None, None
    marked_count, tot_count = 0, 0
    for idx in range(len(tokens) - n_gram + 1):
        random_seed = hash_integer_array_to_int(tokens[idx: idx + n_gram])
        is_marked = random_seed % 2
        if is_marked == 1:
            marked_count += 1
        tot_count += 1
    return compute_z_score(x=marked_count, n=tot_count, p=0.5), marked_count, tot_count - marked_count


@dataclass
class CodeSeg:
    to_detect: str
    with_test: str
    without_test: str

    z_score: float
    marked_count: int
    unmarked_count: int


@dataclass
class Anderson:
    statistic: float
    significance_level: List[float]
    critical_values: List[float]
    dist: str

    @classmethod
    def from_array(cls, arr):
        result = stats.anderson(arr, dist="norm")
        return Anderson(statistic=result.statistic.item(),
                        significance_level=result.significance_level.tolist(),
                        critical_values=result.critical_values.tolist(),
                        dist="norm")
    
    def is_significant(self, level: float=5.):
        for s_level, c_value in zip(self.significance_level, self.critical_values):
            if level == s_level:
                return self.statistic > c_value
        raise RuntimeError(f"Undefined level {level}.")


@dataclass
class ExpRecord:
    norm_anderson: Anderson

    task_id: str
    prompt: str
    test: str

    obfuscated: CodeSeg
    original: CodeSeg
    space: Dict[str, CodeSeg]

    @classmethod
    def from_dict(cls, src) -> "ExpRecord":
        ret = ExpRecord(**src)
        if ret.obfuscated is not None:
            ret.obfuscated = CodeSeg(**ret.obfuscated)
        if ret.original is not None:
            ret.original = CodeSeg(**ret.original)
        ret.space = {k: CodeSeg(**v) for k, v in ret.space.items()}
        return ret


@dataclass
class ExpResult:
    validated_space_count: int
    validated_record_count: int

    original_z_score: float
    obfuscated_z_score: float
    space_exp_z_score: float

    space_norm_rate: float

    records: List[ExpRecord]


deobfuscate_prompt_template = """```
{obfuscated}
```

Here is a normalized JavaScript code segment. Please de-normalize the code. 

Consider there is a code space including all possible de-normalized code. Your job is to select one de-normalized code from the space independently and randomly.
The semantics need to be unchanged.
Instead, change the other properties independently and randomly, including white spaces, variable names, indentation, and with/without comments.

Note:
1. Please infer one version of the original code.
2. You should keep the top-level function name unchanged.

{examples_prompt}

Response format: Your response would only consist of the de-obfuscated code itself.
```"""

examples_prompt_template = """Here are some examples. 
The following codes are acceptable original codes.

{examples}

Please infer the original code other than the above.
"""


def main(request_idx: int,
         request: dict,
         exp_result: ExpResult,
         client_llama: OpenAI,
         client_openai: OpenAI,
         output_file,
         full_model_name,
         deobfuscate_model_name,
         n_gram=5,
         max_tokens=256,
         temperature=1.0,
         run_times=500,
         space_size=30,
         efficiency_threshold=3,
         round_abandon=25,
         round_size=10,
         max_completion_tokens=300,
         de_norm_temperature=1.3,
         few_shot=5) -> bool:

    exp_record = ExpRecord(task_id=request["task_id"],
                           prompt=request["prompt"],
                           test=request["test"],
                           obfuscated=None,
                           original=None,
                           space={},
                           norm_anderson=None)

    logging.info(f"Task {exp_record.task_id} started.")

    exp_result.records.append(exp_record)

    completion = client_llama.completions.create(model=full_model_name,
                                                 prompt=request["prompt"],
                                                 n=run_times,
                                                 temperature=temperature,
                                                 stop="\n}",
                                                 max_tokens=max_tokens)

    logging.info(f"Generation done for {exp_record.task_id}.")

    completions: Dict[str, CodeSeg] = {}

    for choice in completion.choices:
        if choice.finish_reason == "stop" and choice.stop_reason == "\n}":
            code_to_detect = choice.text + "\n}"
            code_w_test = exp_record.prompt + choice.text + "\n}\n\n" + exp_record.test
            code_wo_test = exp_record.prompt + choice.text + "\n}"
            if code_to_detect not in completions:
                z_score, marked_count, unmarked_count = detect(
                    code_to_detect, n_gram)
                if z_score is None:
                    # logging.warning("Detect failed for -\n" + code_to_detect)
                    continue
                completions[code_to_detect] = CodeSeg(to_detect=code_to_detect,
                                                      with_test=code_w_test,
                                                      without_test=code_wo_test,
                                                      z_score=z_score,
                                                      marked_count=marked_count,
                                                      unmarked_count=unmarked_count)
    code_segments: List[CodeSeg] = list(completions.values())
    code_segments.sort(key=lambda c: -c.z_score)
    highest_idx = 0
    for code_segment in code_segments:
        if not test(code_segment.with_test):
            continue
        highest_idx += 1
        if highest_idx >= round_abandon:
            break
        exp_record.original = code_segment
        exp_record.space = {}
        try:
            obfuscated_code = obfuscate(exp_record.prompt, 
                                        exp_record.original.without_test)
            z_score, marked_count, unmarked_count = detect(
                obfuscated_code, n_gram)
            exp_record.obfuscated = CodeSeg(to_detect=obfuscated_code,
                                            with_test=obfuscated_code + "\n\n" + exp_record.test,
                                            without_test=obfuscated_code,
                                            z_score=z_score,
                                            marked_count=marked_count,
                                            unmarked_count=unmarked_count)
        except Exception as e:
            logging.error("%s", e, exc_info=True)
            continue

        for round_idx in range(round_abandon):
            current_space_codes = [c.without_test for c in exp_record.space.values()]
            current_space_codes.append(exp_record.original.without_test)
            if len(current_space_codes) >= efficiency_threshold:
                selected_examples = random.sample(current_space_codes, 
                                                  k=min(few_shot, len(current_space_codes)))
                examples_prompt = examples_prompt_template.format(
                    examples="\n\n".join(selected_examples))
            else:
                examples_prompt = ""
            prompt = deobfuscate_prompt_template.format(
                obfuscated=exp_record.obfuscated.without_test,
                examples_prompt=examples_prompt,
            )

            logging.info(f"Current space size: {len(exp_record.space)}.")

            completion = client_openai.chat.completions.create(
                model=deobfuscate_model_name,
                messages=[{"role": "user", "content": prompt}],
                n=round_size,
                temperature=de_norm_temperature,
                max_completion_tokens=max_completion_tokens
            )
            deobfuscated_codes = []
            for choice in completion.choices:
                message = choice.message.content.strip()
                if "\n```" in message:
                    message = "\n".join(message.split("\n")[1: -1])
                deobfuscated_codes.extend(
                    re.split(r"// De-obfuscated code [0-9]+:", message))
            random.shuffle(deobfuscated_codes)
            for deobfuscated_code in deobfuscated_codes:
                try:
                    deobfuscated_code = deobfuscated_code.strip()
                    obfuscated_code = obfuscate(exp_record.prompt,
                                                deobfuscated_code)
                    if obfuscated_code.strip() == exp_record.obfuscated.without_test.strip():
                        z_score, marked_count, unmarked_count = detect(
                            deobfuscated_code, n_gram)
                        exp_record.space[deobfuscated_code] = CodeSeg(
                            to_detect=deobfuscated_code,
                            with_test=deobfuscated_code + "\n\n" + exp_record.test,
                            without_test=deobfuscated_code,
                            z_score=z_score,
                            marked_count=marked_count,
                            unmarked_count=unmarked_count)
                        if len(exp_record.space) >= space_size:
                            break
                except Exception as e:
                    pass
            if len(exp_record.space) >= space_size:
                break
            if len(exp_record.space) < efficiency_threshold:
                break

        if len(exp_record.space) < space_size:
            continue
        else:
            space_codes = list(exp_record.space.keys())
            selected_space_codes = random.sample(space_codes, space_size)
            exp_record.space = {k: exp_record.space[k] for k in selected_space_codes}

        logging.info(f"De-obfuscation done for {exp_record.task_id}. " +
                    f"Group size: {len(exp_record.space)}.")

        def mean_fn(a): return sum(a) / len(a)

        exp_result.validated_record_count = 0
        exp_result.validated_space_count = 0
        obfuscated_z_scores = []
        original_z_scores = []
        space_exp_z_scores = []
        space_norm_count = 0
        for r in exp_result.records:
            if r.original is not None:
                exp_result.validated_record_count += 1
                obfuscated_z_scores.append(r.obfuscated.z_score)
                original_z_scores.append(r.original.z_score)
                exp_result.obfuscated_z_score = mean_fn(obfuscated_z_scores)
                exp_result.original_z_score = mean_fn(original_z_scores)
                if len(r.space) >= space_size:
                    exp_result.validated_space_count += 1
                    space_current_z_scores = [c.z_score for c in r.space.values()]
                    space_exp_z_scores.append(mean_fn(space_current_z_scores))
                    exp_result.space_exp_z_score = mean_fn(space_exp_z_scores)

                    exp_record.norm_anderson = Anderson.from_array(space_current_z_scores)

                    if not exp_record.norm_anderson.is_significant():
                        space_norm_count += 1
                    exp_result.space_norm_rate = space_norm_count / exp_result.validated_space_count

        exp_result.records.sort(key=lambda r: r.task_id)

        with open(output_file, "w") as file:
            file.write(json.dumps(asdict(exp_result), indent=4))

        logging.info(json.dumps(indent=4, obj={
            "original_z_score": exp_result.original_z_score,
            "obfuscated_z_score": exp_result.obfuscated_z_score,
            "space_exp_z_score": exp_result.space_exp_z_score,
            "space_norm_rate": exp_result.space_norm_rate,
        }))

        time_used = int(time.time() - start_time)
        time_remained = int(time_used / (request_idx + 1)
                            * (len(data) - request_idx - 1))
        logging.info(f"Task {exp_record.task_id} done ({request_idx + 1} / {len(data)}). " +
                    f"Time used: {time_used//60}min{time_used%60}sec. " +
                    f"Time remained: {time_remained//60}min{time_remained%60}sec.")

        return True
    return False


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    output_file = "data/ng_ind.json"
    full_model_name = "meta-llama/Llama-3.1-8B-Instruct"
    deobfuscate_model_name = "gpt-4o"
    # deobfuscate_model_name = "deepseek-chat"
    tokenizer = AutoTokenizer.from_pretrained(full_model_name)

    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client_llama = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    client_openai = OpenAI(
        # api_key=os.environ.get("DEEPSEEK_API_KEY"),
        # base_url="https://api.deepseek.com"
    )

    exp_result = ExpResult(validated_space_count=None,
                           validated_record_count=None,
                           original_z_score=None,
                           obfuscated_z_score=None,
                           space_exp_z_score=None,
                           space_norm_rate=None,
                           records=[])

    if os.path.exists(output_file):
        with open(output_file) as file:
            cache = json.loads("".join(file.readlines()))
        for record_dict in cache["records"]:
            exp_result.records.append(ExpRecord.from_dict(record_dict))

    cached_task_ids = set([r.task_id for r in exp_result.records])

    with open("data/original/humaneval-x_js.jsonl") as file:
        data = [json.loads(line) for line in file]
        data = [t for t in data if t["task_id"] not in cached_task_ids]

    start_time = time.time()
    for request_idx, request in enumerate(data):
        main(request_idx=request_idx, request=request, exp_result=exp_result,
             client_llama=client_llama, client_openai=client_openai,
             output_file=output_file, full_model_name=full_model_name,
             deobfuscate_model_name=deobfuscate_model_name)

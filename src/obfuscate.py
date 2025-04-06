import torch
from transformers import (
    pipeline,
    StoppingCriteria,
    SynthIDTextWatermarkingConfig,
    StoppingCriteriaList,
    LogitsProcessorList
)
import subprocess
import sys
import logging
import json
from dataclasses import asdict
sys.path.append("src")

from _data_structure import *
from _sweet import (
    SweetLogitsProcessor,
    WatermarkLogitsProcessor,
)
from _util import (
    create_tempdir,
    dataclass_2_str
)
from _hf_obj import (
    get_hf_pipeline, 
    get_hf_tokenizer,
    get_synthid_config
)
from evaluation import evaluate
from detection import detect


def obfuscate(task: ObfTask, gen_task: GenTask):
    logging.info(f"[start] {dataclass_2_str(task)}")

    assert all(o is not None for o in [
        gen_task.p4d, gen_task.g4d, gen_task.solution])
    with create_tempdir() as work_dir:
        if gen_task.language == "js":
            solution_file_name = "solution.js"
            obf_file_name = "obf.js"
            with open(solution_file_name, "w") as file:
                file.write(gen_task.solution)
            if task.obf_name == "javascript-obfuscator":
                cmd = [
                    "uglifyjs", "-c", "-b",
                    "-m", f"reserved=[{gen_task.entry_point}]",
                    "--", solution_file_name, ">", obf_file_name
                ]
            elif task.obf_name == "uglifyjs":
                cmd = [
                    "javascript-obfuscator", solution_file_name,
                    "--reserved-names", f"\"{gen_task.entry_point}\"",
                    "--output", obf_file_name,
                    "--identifier-names-generator", "mangled-shuffled"
                ]
            else:
                raise NotImplementedError()
            subprocess.run(cmd, timeout=10, capture_output=True)
            with open(obf_file_name) as file:
                task.solution = "".join(file.readlines())
            task.p4d = gen_task.split("*/")[0] + "*/\n"
            task.g4d = task.solution
        elif gen_task.language == "py":
            solution_file_name = "solution.py"
            obf_file_name = "obf.py"
            with open(solution_file_name, "w") as file:
                file.write(gen_task.solution)
            if task.obf_name == "pyminify":
                cmd = [
                    "pyminify", "--remove-literal-statements",
                    solution_file_name, ">", obf_file_name, "2>&1"
                ]
            else:
                raise NotImplementedError()
            subprocess.run(cmd, timeout=10, capture_output=True)
            with open(obf_file_name) as file:
                task.solution = "".join(file.readlines())
            obfuscated_code_lines = task.solution.split("\n")
            inline_function_content = ""
            for idx, line in enumerate(obfuscated_code_lines):
                if task.entry_point in line:
                    inline_function_content = "):".join(line.split("):")[1:])
                    if len(inline_function_content) > 0:
                        inline_function_content = inline_function_content.strip()
                        inline_function_content = f"    {inline_function_content}\n"
                    break
            task.p4d = gen_task.p4d
            task.g4d = inline_function_content + "\n".join(obfuscated_code_lines[idx + 1:])
        else:
            raise NotImplementedError()
    logging.info(f"[obfuscated] {dataclass_2_str(task)}")
    
    evaluate(task=task, gen_task=gen_task)
    logging.info(f"[evaluated] {dataclass_2_str(task)}")

    detect(task=task, gen_task=gen_task)
    logging.info(f"[detected] {dataclass_2_str(task)}")

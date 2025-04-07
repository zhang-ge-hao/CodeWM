import torch
from transformers import (
    pipeline,
    StoppingCriteria,
    SynthIDTextWatermarkingConfig,
    StoppingCriteriaList,
    LogitsProcessorList
)
import subprocess
import traceback
import sys
import logging
import json
from dataclasses import asdict
sys.path.append("src")

from _dataclass import *
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

    tokenizer = get_hf_tokenizer(gen_task.model_name)

    with create_tempdir() as work_dir:
        if gen_task.language == "js":
            solution_file_name = "solution.js"
            with open(solution_file_name, "w") as file:
                file.write(gen_task.solution)
            if task.obf_name == "uglifyjs":
                cmd = [
                    "uglifyjs", "-c", "-b",
                    "-m", f"reserved=[{gen_task.entry_point}]",
                    "--", solution_file_name
                ]
                result = subprocess.run(cmd, timeout=10, capture_output=True, text=True)
                if result.returncode != 0:
                    logging.error(f"Error during obfuscation:\n{result.stderr}")
                else:
                    task.solution = result.stdout
                    task.s_len = len(tokenizer.encode(task.solution))
            elif task.obf_name == "javascript-obfuscator":
                obf_file_name = "obf.js"
                cmd = [
                    "javascript-obfuscator", solution_file_name,
                    "--reserved-names", f"\"{gen_task.entry_point}\"",
                    "--output", obf_file_name,
                    "--identifier-names-generator", "mangled-shuffled"
                ]
                subprocess.run(cmd, timeout=10, capture_output=True)
                with open(obf_file_name) as file:
                    task.solution = "".join(file.readlines())
                    task.s_len = len(tokenizer.encode(task.solution))
            else:
                raise NotImplementedError()
            if gen_task.is_inst:
                task.p4d = gen_task.p4d
                task.g4d = task.solution
            else:
                task.p4d = gen_task.p4d.split("*/")[0] + "*/\n"
                task.g4d = task.solution
        elif gen_task.language == "py":
            solution_file_name = "solution.py"
            with open(solution_file_name, "w") as file:
                file.write(gen_task.solution)
            if task.obf_name == "pyminify":
                cmd = ["pyminify", "--remove-literal-statements", 
                       solution_file_name]
            elif task.obf_name == "pyminifier":
                cmd = ["pyminifier", solution_file_name]
            else:
                raise NotImplementedError()
            try:
                result = subprocess.run(cmd, timeout=10, capture_output=True, text=True)
                if result.returncode != 0:
                    logging.error(f"Error during obfuscation:\n{result.stderr}")
                else:
                    task.solution = result.stdout
                    # remove comment line from pyminifier
                    lines = task.solution.split("\n")
                    lines = [l for l in lines if "# Created by pyminifier" not in l]
                    task.solution = "\n".join(lines)
                    task.s_len = len(tokenizer.encode(task.solution))
            except Exception as e:
                logging.error(traceback.format_exc())

            if gen_task.is_inst:
                task.p4d = gen_task.p4d
                task.g4d = task.solution
            else:
                obfuscated_code_lines = task.solution.split("\n")
                inline_function_content = ""
                for idx, line in enumerate(obfuscated_code_lines):
                    if gen_task.entry_point in line:
                        inline_function_content = "):".join(line.split("):")[1:])
                        if len(inline_function_content) > 0:
                            inline_function_content = inline_function_content.strip()
                            inline_function_content = f"    {inline_function_content}\n"
                        break
                task.p4d = gen_task.p4d
                task.g4d = inline_function_content + "\n".join(
                    obfuscated_code_lines[idx + 1:])
        else:
            raise NotImplementedError()
    logging.info(f"[obfuscated] {dataclass_2_str(task)}")
    
    evaluate(task=task, gen_task=gen_task)
    task.bad_trans = task.passed != gen_task.passed
    logging.info(f"[evaluated] {dataclass_2_str(task)}")

    detect(task=task, gen_task=gen_task)
    logging.info(f"[detected] {dataclass_2_str(task)}")

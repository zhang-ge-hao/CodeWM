from typing import List, Dict
from dataclasses import dataclass


@dataclass
class GenTask:
    id: str
    task_name: str
    model_name: str
    watermarking: str

    language: str
    is_inst: bool
    need_obf: bool

    temperature: float
    delta: float
    gamma: float
    entropy_threshold: float

    ori_prompt: str
    entry_point: str
    test: str

    p4d: str # prompt for detection
    g4d: str # generation for detection
    solution: str

    passed: bool
    res_d: float # detection result


@dataclass
class ObfTask:
    id: str
    gen_task_id: str

    obf_name: str

    p4d: str # prompt for detection
    g4d: str # generation for detection
    solution: str

    passed: bool
    res_d: float # detection result

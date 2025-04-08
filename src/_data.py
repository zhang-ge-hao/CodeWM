from typing import List, Dict
from dataclasses import dataclass


@dataclass
class GenTask:
    id: str
    task_name: str
    dataset_name: str
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
    s_len: int

    passed: bool
    z_score: float # detection result
    p_value: float

    custom_seed: int


@dataclass
class ObfTask:
    id: str
    gen_task_id: str

    obf_name: str

    p4d: str # prompt for detection
    g4d: str # generation for detection
    solution: str
    s_len: int # token length of solution

    passed: bool
    z_score: float # detection result
    p_value: float

    bad_trans: bool


@dataclass
class DataPoint:
    obf_name: str
    pass1: float
    auroc: float

    language: str
    dataset_name: str
    model_name: str
    watermarking: str

    z_score: float # mean z_score
    p_value: float # mean p_value
    exp_c: int # expected case count
    comp_c: int # compilable case count
    len_sum: int # length sum compilable cases

    temperature: float
    delta: float
    gamma: float
    entropy_threshold: float

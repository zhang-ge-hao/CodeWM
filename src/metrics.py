import sys, os
from dataclasses import asdict
from typing import List, Dict
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_curve, auc
sys.path.append("src")

from _dataclass import *


def cal_ideal_auroc(samples: List[float]) -> float:
    samples = np.array(samples)
    negative_dist = norm(0, 1)
    thresholds = np.linspace(-15, 15, 3000)
    tpr = [np.mean(samples > t) for t in thresholds]
    fpr = 1 - negative_dist.cdf(thresholds)
    auroc = auc(fpr, tpr)
    return auroc


def calculate(gen_tasks: List[GenTask], 
              obf_tasks: List[ObfTask]) -> List[DataPoint]:
    if len(gen_tasks) == 0:
        return []
    
    assert all(t.need_obf for t in gen_tasks) or \
        all(not t.need_obf for t in gen_tasks)
    
    need_obf = gen_tasks[0].need_obf

    assert (need_obf and len(obf_tasks) > 0) or \
        (not need_obf and len(obf_tasks == 0))

    for field_name in ["language", "dataset_name", 
                       "model_name", "watermarking"]:
        assert len(set([asdict(t)[field_name] for t in gen_tasks])) == 1
    
    language = gen_tasks[0].language
    dataset_name = gen_tasks[0].dataset_name
    model_name = gen_tasks[0].model_name
    watermarking = gen_tasks[0].watermarking

    if need_obf:
        obf_name_2_success_gen_task_ids: Dict[str, set] = {}
        for obf_task in obf_tasks:
            if obf_task.obf_name not in obf_name_2_success_gen_task_ids:
                obf_name_2_success_gen_task_ids[obf_task.obf_name] = set()
            if obf_task.solution is not None:
                obf_name_2_success_gen_task_ids[obf_task.obf_name].add(
                    obf_task.gen_task_id)
        retained_gen_task_ids = set([t.id for t in gen_tasks])
        for ids in obf_name_2_success_gen_task_ids.values():
            retained_gen_task_ids &= ids
    else:
        retained_gen_task_ids = set([t.id for t in gen_tasks])
    
    obf_name_2_tasks: Dict[str, list] = {
        "Original": [t for t in gen_tasks if t.id in retained_gen_task_ids]}
    
    if need_obf:
        for obf_task in obf_tasks:
            if obf_task.gen_task_id in retained_gen_task_ids:
                if obf_task.obf_name not in obf_name_2_tasks:
                    obf_name_2_tasks[obf_task.obf_name] = []
                obf_name_2_tasks[obf_task.obf_name].append(obf_task)

    ret: List[DataPoint] = []
    for obf_name, tasks in obf_name_2_tasks.items():
        assert all(isinstance(t, GenTask) or isinstance(t, ObfTask) for t in tasks)
        pass1 = sum([(1 if t.passed else 0) for t in tasks]) / len(tasks)
        auroc = cal_ideal_auroc([t.z_score for t in tasks])
        z_score = sum([t.z_score for t in tasks]) / len(tasks)
        p_value = sum([t.p_value for t in tasks]) / len(tasks)
        len_sum = sum([t.s_len for t in tasks])

        bad_trans = None
        if obf_name != "Original":
            bad_trans = sum([(1 if t.bad_trans else 0) for t in tasks])

        dp = DataPoint(language=language, dataset_name=dataset_name,
                       model_name=model_name, watermarking=watermarking,
                       obf_name=obf_name, pass1=pass1, auroc=auroc,
                       z_score=z_score, p_value=p_value, exp_c=len(gen_tasks),
                       comp_c=len(tasks), len_sum=len_sum, bad_trans=bad_trans)
        ret.append(dp)
    return ret
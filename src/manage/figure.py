import os
import json
import sys
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("src")
from _data import *

# style_key_2_style_values = {
#     "color": ["green", "red", "blue", "orange", "purple"],
#     "marker": ["o", "s", "^", "D", "p"],
#     "edgecolor": ["green", "red", "blue", "orange", "purple"],
# }


def is_idx(folder_name):
    try:
        int(folder_name)
        return True
    except:
        return False


def get_color(dp: DataPoint):
    if dp.watermarking not in ["sweet", "wllm"]:
        return "white"
    delta_2_color = {0.5: "green", 1.0: "red",
                     2.0: "blue", 3.0: "orange",
                     4.0: "purple"}
    return delta_2_color[dp.delta]


def get_marker(dp: DataPoint):
    if dp.watermarking not in ["sweet", "wllm"]:
        return "o"
    gamma_2_marker = {0.10: "o", 0.25: "s", 0.50: "^"}
    return gamma_2_marker[dp.gamma]


def get_edgecolor(dp: DataPoint):
    if dp.watermarking == "wllm":
        return "grey"
    temp_2_edgecolor = {0.25: "green", 0.5: "red", 
                        0.75: "blue", 1.0: "orange",
                        1.25: "purple"}
    if dp.watermarking == "synthid":
        return temp_2_edgecolor[dp.temperature]
    entropy_threshold_2_edgecolor = {0.3: "green", 0.6: "red", 
                                     0.9: "blue", 1.2: "orange",}
    return entropy_threshold_2_edgecolor[dp.entropy_threshold]


def draw_plot(dps: List[DataPoint], figure_path, no_wm_res,
              threshold_ratio=0.8):
    plt.rcParams.update({
        'font.size': 12,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times"], 
    })
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.grid(True, which='major', 
            linestyle='--', linewidth=0.3, color="#A0A0A0")
    ax.axhline(y=0.5, color='orange', 
               linestyle='--', linewidth=2)
    ax.axvline(x=no_wm_res * threshold_ratio, color='blue', 
               linestyle='--', linewidth=2)
    for dp in dps:
        ax.scatter(
            x=dp.pass1, 
            y=dp.auroc, 
            color=get_color(dp), 
            marker=get_marker(dp),
            edgecolors=get_edgecolor(dp), 
            s=50,
            linewidths=1.5)
    # ax.set_xticks(np.arange(0, 1, 0.05))
    # ax.set_yticks(np.arange(0, 1, 0.1))

    ax.set_ylim(0.2, 1.05)
    # ax.set_xlim(0.05, 1.0)

    plt.savefig(figure_path)


NO_WM = {}

def get_no_wm_res(mn, dn, temp):
    if len(NO_WM) == 0:
        result_root = "data/result"
        for figure_folder_name in os.listdir(result_root):
            if not "no_wm" in figure_folder_name:
                continue
            figure_root = f"{result_root}/{figure_folder_name}"
            dp_idxs = os.listdir(figure_root)
            dp_idxs = [i for i in dp_idxs if is_idx(i)]
            obf_name_2_dps: Dict[str, List[DataPoint]] = {}
            for dp_idx in dp_idxs:
                metric_path = f"{figure_root}/{dp_idx}/metrics.jsonl"
                if os.path.exists(metric_path):
                    with open(metric_path) as file:
                        dps = [DataPoint(**json.loads(l)) for l in file]
                    for dp in dps:
                        if dp.watermarking == "no_wm":
                            NO_WM[(dp.model_name, 
                                dp.dataset_name, 
                                dp.temperature)] = dp.pass1
    return NO_WM[(mn, dn, temp)]

WLLM = {}

def get_wllm_res(mn, dn, obf):
    if len(WLLM) == 0:
        result_root = "data/result"
        for figure_folder_name in os.listdir(result_root):
            if not "wllm" in figure_folder_name:
                continue
            figure_root = f"{result_root}/{figure_folder_name}"
            dp_idxs = os.listdir(figure_root)
            dp_idxs = [i for i in dp_idxs if is_idx(i)]
            obf_name_2_dps: Dict[str, List[DataPoint]] = {}
            for dp_idx in dp_idxs:
                metric_path = f"{figure_root}/{dp_idx}/metrics.jsonl"
                if os.path.exists(metric_path):
                    with open(metric_path) as file:
                        dps = [DataPoint(**json.loads(l)) for l in file]
                    for dp in dps:
                        if dp.watermarking == "wllm":
                            key = (dp.model_name,
                                   dp.dataset_name, 
                                   dp.obf_name)
                            if key not in WLLM:
                                WLLM[key] = []
                            WLLM[key].append(dp)
    return WLLM[(mn, dn, obf)]

if __name__ == "__main__":
    result_root = "data/result"
    figure_output_root = "data/figure"
    for figure_folder_name in os.listdir(result_root):
        if "no_wm" in figure_folder_name or "wllm" in figure_folder_name:
            continue
        figure_root = f"{result_root}/{figure_folder_name}"
        dp_idxs = os.listdir(figure_root)
        dp_idxs = [i for i in dp_idxs if is_idx(i)]
        obf_name_2_dps: Dict[str, List[DataPoint]] = {}
        for dp_idx in dp_idxs:
            metric_path = f"{figure_root}/{dp_idx}/metrics.jsonl"
            if os.path.exists(metric_path):
                with open(metric_path) as file:
                    dps = [DataPoint(**json.loads(l)) for l in file]
                for dp in dps:
                    if dp.obf_name not in obf_name_2_dps:
                        obf_name_2_dps[dp.obf_name] = []
                    obf_name_2_dps[dp.obf_name].append(dp)
        figure_output_dir = f"{figure_output_root}/{figure_folder_name}"
        for obf_name, dps in obf_name_2_dps.items():
            dp_demo = dps[0]
            if dp_demo.watermarking == "sweet":
                wllm_dps = get_wllm_res(dp_demo.model_name, 
                                        dp_demo.dataset_name, 
                                        dp_demo.obf_name)
                dps = wllm_dps + dps
            figure_path = f"{figure_output_dir}/{obf_name}.pdf"
            if dp_demo.watermarking != "synthid":
                no_wm_res = get_no_wm_res(dp_demo.model_name, 
                    dp_demo.dataset_name, dp_demo.temperature)
                draw_plot(dps, figure_path, no_wm_res)

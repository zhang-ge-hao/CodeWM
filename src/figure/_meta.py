import os
import json
import sys
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import matplotlib.patheffects as path_effects
from matplotlib.axes import Axes
from itertools import product
from matplotlib.lines import Line2D
sys.path.append("src")
from _data import *

dark_set = [
    "#d20962", 
    "#f47721", 
    "#7ac143", 
    "#00a78e", 
    "#00bce4", 
    "#7d3f98"
]
light_set = [
    "#744da8",
    "#1fb3e0",
    "#49c219",
    "#eeb417",
    "#d65129"
]
classical_set = [
    "#00a3e2",
    "#1ba548",
    "#8a90c7",
    "#f1860e",
    "#e41b13"
]
rgba_alpha = 0.5
rgba_set = [
    (0.0, 0.6392156862745098, 0.8862745098039215, rgba_alpha), 
    (0.10588235294117647, 0.6470588235294118, 0.2823529411764706, rgba_alpha), 
    (0.5411764705882353, 0.5647058823529412, 0.7803921568627451, rgba_alpha), 
    (0.9450980392156862, 0.5254901960784314, 0.054901960784313725, rgba_alpha), 
    (0.8941176470588236, 0.10588235294117647, 0.07450980392156863, rgba_alpha)
]
wllm_edgecolor = "#44403f"

dataset_name_map = {
    "humaneval_py": "HumanEval",
    "humaneval_js": "HumanEval-X-JS",
    "mbpp_py": "MBPP+ (Base)",
    "mbpp_js": "MBPP-JS",
}
obf_name_map = {
    "pyminify": "Python-Minifier",
    "pyminifier": "PyMinifier",
    "javascript-obfuscator": "JS Obfuscator",
    "uglifyjs": "UglifyJS",
}
model_name_map = {
    "DSCoderBase33B": "DeepSeek Coder 33B Base",
    "Llama31Instruct8B": "LLaMA 3.1 8B Instruct"
}
lang_map = {
    "js": "JavaScript",
    "py": "Python"
}
ref_line_config = {
    "linestyle": (1, (3, 3)), 
    "linewidth": 1.5,
}
h_line_color = "pink"
v_line_color = "lightblue"
deltas = [0.5, 1.0, 2.0, 3.0, 4.0]
delta_2_color = {k: v for k, v in zip(deltas, rgba_set)}
temps = [0.25, 0.5, 0.75, 1.0, 1.25]
temp_2_edgecolor = {k: v for k, v in zip(temps, classical_set)}
entropy_thresholds = [0.3, 0.6, 0.9, 1.2]
entropy_threshold_2_edgecolor = {k: v for k, v in zip(
    entropy_thresholds, classical_set)}
gamma_2_marker = {0.10: "o", 0.25: "s", 0.50: "^"}

marker_set = ["o", "s", "^", "D", "p"]

def is_idx(folder_name):
    try:
        int(folder_name)
        return True
    except:
        return False


def add_gamma_legend(ax: Axes):
    ret = []
    for gamma, marker in gamma_2_marker.items():
        ret.append(
            Line2D([0], [0], 
                   marker=marker, 
                   color='darkgrey', 
                   label=f'$\gamma$ = {gamma:.2f}',
                   markerfacecolor='none', 
                   markersize=4,
                   linewidth=2,
                   linestyle='none')
        )
    legend = ax.legend(handles=ret, ncol=1, title='Style',
                       loc='upper right', title_fontsize=7, 
                       fontsize=7, frameon=True,
                       bbox_to_anchor=(0.74, 0.99))
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.4)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(0.5)
    legend.get_title().set_ha('left')
    legend._legend_box.align = "left"
    ax.add_artist(legend)


def add_delta_legend(ax: Axes):
    ret = []
    for delta, color in delta_2_color.items():
        ret.append(
            Line2D([0], [0], 
                   marker="s", 
                   label=f'$\delta$ = {delta:.2f}',
                   markerfacecolor=color, 
                   markeredgecolor="none",
                   markersize=4,
                   linewidth=2,
                   linestyle='none')
        )
    legend = ax.legend(handles=ret, ncol=1, title='Filling Color',
                       loc='upper right', title_fontsize=7, 
                       fontsize=7, frameon=True,
                       bbox_to_anchor=(0.49, 0.99))
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.4)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(0.5)
    legend.get_title().set_ha('left')
    legend._legend_box.align = "left"
    ax.add_artist(legend)


def add_entropy_threshold_legend(ax: Axes):
    ret = []
    for entropy_threshold, color in list(
            entropy_threshold_2_edgecolor.items()) + [(0, wllm_edgecolor)]:
        label = f'$\\tau$ = {entropy_threshold:.2f}'
        ret.append(
            Line2D([0], [0], 
                   marker="o", 
                   color=color, 
                   label=label,
                   markerfacecolor='none', 
                   markersize=4,
                   linewidth=2,
                   linestyle='none')
        )
    legend = ax.legend(handles=ret, ncol=1, title='Border Color',
                       loc='upper right', title_fontsize=7, 
                       fontsize=7, frameon=True,
                       bbox_to_anchor=(0.99, 0.99))
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.4)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(0.5)
    legend.get_title().set_ha('left')
    legend._legend_box.align = "left"
    ax.add_artist(legend)


def get_color(dp: DataPoint):
    if dp.watermarking not in ["sweet", "wllm"]:
        return "white"
    return delta_2_color[dp.delta]


def get_marker(dp: DataPoint):
    if dp.watermarking not in ["sweet", "wllm"]:
        return "o"
    return gamma_2_marker[dp.gamma]


def get_edgecolor(dp: DataPoint):
    if dp.watermarking == "wllm":
        return wllm_edgecolor
    if dp.watermarking == "synthid":
        return temp_2_edgecolor[dp.temperature]
    return entropy_threshold_2_edgecolor[dp.entropy_threshold]

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

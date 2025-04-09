import os
import json
import sys
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import matplotlib.patheffects as path_effects
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from itertools import product
sys.path.append("src")
from _data import *
from figure.g_r_main import *


if __name__ == "__main__":
    result_root = "data/result"
    figure_output_root = "data/figure"

    selected_model = "DSCoderBase33B"
    lang = "js"
    selected_obf = "javascript-obfuscator"
    selected_ds = "mbpp_js"
    plt.rcParams.update({
        'font.size': 8,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times"], 
    })

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(4, 5.5),
                            sharex="col")
    def ff_name_2_col(ff_name):
        for r_id, d in enumerate(dataset_ordered):
            if d in ff_name:
                return r_id
    def obf_name_2_row(obf_name):
        if obf_name == "Original":
            return 0
        else:
            return 1

    for figure_folder_name in os.listdir(result_root):
        if "no_wm" in figure_folder_name or "wllm" in figure_folder_name:
            continue
        if selected_model not in figure_folder_name:
            continue
        if selected_ds not in figure_folder_name:
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
        for obf_name, dps in obf_name_2_dps.items():
            dp_demo = dps[0]
            if dp_demo.watermarking != "sweet":
                continue
            if dp_demo.obf_name not in [selected_obf, "Original"]:
                continue
            wllm_dps = get_wllm_res(dp_demo.model_name, 
                                    dp_demo.dataset_name, 
                                    dp_demo.obf_name)
            dps = wllm_dps + dps

            no_wm_res = get_no_wm_res(dp_demo.model_name, 
                dp_demo.dataset_name, dp_demo.temperature)
            ax_row = obf_name_2_row(obf_name)
            ax = axs[ax_row]
            draw_plot(ax, dps, no_wm_res, 
                        add_x_label=True,
                        add_y_label=True, 
                        add_h_line_tag=True, 
                        dataset_name=dp_demo.dataset_name)
            if ax_row == 1:
                add_gamma_legend(ax)
                add_delta_legend(ax)
                add_entropy_threshold_legend(ax)
            if ax_row == 0:
                ax.set_title(
                    f"Green-Red Watermarks vs. {obf_name_map[selected_obf]}", 
                    fontsize=11, pad=13)
            ax.set_ylabel(f"AUROC")
    plt.tight_layout()
    plt.savefig(f"{figure_output_root}/g_r_intro--{selected_model}--{lang}.pdf")

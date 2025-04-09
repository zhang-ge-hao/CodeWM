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
sys.path.append("src")
from _data import *
from figure._meta import *


def draw_plot(ax: Axes, dps: List[DataPoint], no_wm_res, 
              add_x_label, add_y_label, dataset_name, 
              threshold_ratio=0.8,):
    global GLOBAL_ZORDER
    GLOBAL_ZORDER = 0
    def zo():
        global GLOBAL_ZORDER
        GLOBAL_ZORDER += 1
        return GLOBAL_ZORDER

    dataset_name = dataset_name_map[dataset_name]

    # Grid and Ticks
    major_gird_config = {
        "linestyle": (0, (0.5, 2)), 
        "linewidth": 0.5, 
        "color": "#a0a0a0"
    }
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(True, which='major', **major_gird_config,
            zorder=zo())
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.grid(True, which='minor', **major_gird_config,
            zorder=zo())

    # Points
    for dp in dps:
        ax.scatter(
            x=dp.pass1, 
            y=dp.auroc, 
            color=get_color(dp), 
            marker=get_marker(dp),
            edgecolors=get_edgecolor(dp), 
            s=20,
            linewidths=0.6,
            zorder=zo())

    # Range
    ax.set_ylim(0.2, 1.05)

    if dps[0].obf_name == "Original":
        x_label = "Original WM"
    else:
        x_label = obf_name_map[dps[0].obf_name] + " Obfuscated"
    if add_x_label:
        ax.set_xlabel(f"Pass@1\n({x_label})")
    else:
        ax.tick_params(labelbottom=False)

    if add_y_label:
        ax.set_ylabel(f"{dataset_name}\nAUROC")
    else:
        ax.tick_params(labelleft=False)


if __name__ == "__main__":
    result_root = "data/result"
    figure_output_root = "data/figure"

    models_ordered = ["DSCoderBase33B", "Llama31Instruct8B"]
    lang_2_datasets = {
        "py": ["humaneval_py", "mbpp_py"],
        "js": ["humaneval_js", "mbpp_js"],
    }

    plt.rcParams.update({
        'font.size': 8,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times"], 
    })

    for model_name, lang in product(models_ordered, lang_2_datasets.keys()):
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(8, 4),
                                sharey="row", sharex="col")
        dataset_ordered = lang_2_datasets[lang]
        def ff_name_2_row(ff_name):
            for r_id, d in enumerate(dataset_ordered):
                if d in ff_name:
                    return r_id
        def obf_name_2_col(obf_name):
            if obf_name == "Original":
                return 0
            elif obf_name in ["pyminify", "javascript-obfuscator"]:
                return 1
            elif obf_name in ["pyminifier", "uglifyjs"]:
                return 2

        for figure_folder_name in os.listdir(result_root):
            if "no_wm" in figure_folder_name or "wllm" in figure_folder_name:
                continue
            if model_name not in figure_folder_name:
                continue
            if all(dsn not in figure_folder_name for dsn in dataset_ordered):
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
                wllm_dps = get_wllm_res(dp_demo.model_name, 
                                        dp_demo.dataset_name, 
                                        dp_demo.obf_name)
                dps = wllm_dps + dps

                no_wm_res = get_no_wm_res(dp_demo.model_name, 
                    dp_demo.dataset_name, dp_demo.temperature)
                ax_row = ff_name_2_row(figure_folder_name)
                ax_col = obf_name_2_col(obf_name)
                ax = axs[ax_row, ax_col]
                draw_plot(ax, dps, no_wm_res, 
                          add_x_label=ax_row == len(dataset_ordered) - 1,
                          add_y_label=ax_col == 0, 
                          dataset_name=dp_demo.dataset_name)
        fig.suptitle(f"{model_name_map[model_name]} vs. {lang_map[lang]}", 
                     fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{figure_output_root}/g_r_app--{model_name}--{lang}.pdf")

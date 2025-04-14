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
              add_title,
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
        title = "Original WM"
    else:
        title = obf_name_map[dps[0].obf_name] + " Obfuscated"
    if add_title:
        ax.set_title(title)
    if add_x_label:
        ax.set_xlabel(f"Pass@1")
    else:
        ax.tick_params(labelbottom=False)

    if add_y_label:
        ax.set_ylabel(f"{dataset_name}\nAUROC")
    else:
        ax.tick_params(labelleft=False)


if __name__ == "__main__":
    result_root = "data/result"
    figure_output_root = "data/figure"

    model_name = "Llama31Instruct8B"
    
    # dataset_ordered = ["humaneval_py", "mbpp_py", "humaneval_js", "mbpp_js"]

    langs = "js"

    dataset_name = "humaneval_js"

    plt.rcParams.update({
        'font.size': 8,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times"], 
    })

    selected_obf_names = ["Original", "javascript-obfuscator"]

    for ngram_len in range(2, 6):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(4, 1.6),
                            sharey="all", sharex="all")
        for ax_col, selected_obf_name in enumerate(selected_obf_names):
            for figure_folder_name in os.listdir(result_root):
                if "wllm" not in figure_folder_name:
                    continue
                if model_name not in figure_folder_name:
                    continue
                if dataset_name not in figure_folder_name:
                    continue
                figure_root = f"{result_root}/{figure_folder_name}"
                dp_idxs = os.listdir(figure_root)
                dp_idxs = [i for i in dp_idxs if is_idx(i)]
                tot_dps: List[DataPoint] = []
                for dp_idx in dp_idxs:
                    metric_path = f"{figure_root}/{dp_idx}/metrics.jsonl"
                    if os.path.exists(metric_path):
                        with open(metric_path) as file:
                            dps = [DataPoint(**json.loads(l)) for l in file]
                        for dp in dps:
                            if dp.obf_name != selected_obf_name:
                                continue
                            if dp.ngram_len != ngram_len:
                                continue
                            tot_dps.append(dp)
                dp_demo = tot_dps[0]
                no_wm_res = get_no_wm_res(dp_demo.model_name, 
                    dp_demo.dataset_name, dp_demo.temperature)
                ax = axs[ax_col]
                
                sum_pass1 = sum([dp.pass1 for dp in tot_dps])
                sum_auroc = sum([dp.auroc for dp in tot_dps])
                print(f"{ngram_len} {selected_obf_name}")
                print(f"{sum_pass1 / len(tot_dps):.3f} {sum_auroc / len(tot_dps):.3f}")

                draw_plot(ax, tot_dps, no_wm_res, 
                            add_title=True,
                            add_x_label=True,
                            add_y_label=ax_col == 0, 
                            dataset_name=dp_demo.dataset_name)
            # fig.suptitle(f"{ngram_len}-grams", 
            #              fontsize=11, y=0.93)
            plt.tight_layout()
        plt.savefig(f"{figure_output_root}/wllm_ngram--{ngram_len}-gram.pdf")

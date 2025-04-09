import os
import json
import sys
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import matplotlib.patheffects as path_effects
from matplotlib.axes import Axes
from itertools import product
sys.path.append("src")
from _data import *
from figure._meta import *


def draw_plot(ax: Axes, obf_dps: List[DataPoint], ori_dps: List[DataPoint], 
              add_x_label, add_left_y_label, add_right_y_label,
              add_h_line_tag, model_name, obf_name, dataset_name):
    ori_dps.sort(key=lambda dp: dp.temperature)
    obf_dps.sort(key=lambda dp: dp.temperature)
    assert len(ori_dps) == len(obf_dps) and \
        all(dp1.temperature == dp2.temperature \
            for dp1, dp2 in zip(ori_dps, obf_dps))

    # add right Y
    ax2 = ax.twinx()
    # pass1
    ax2.plot([dp.temperature for dp in obf_dps], 
            [dp.pass1 for dp in obf_dps], 'purple',
            linestyle='--', linewidth=1.5)
    # ori auroc
    ax.plot([dp.temperature for dp in obf_dps], 
             [dp.auroc for dp in ori_dps], "#0863b5",
             marker="o", linewidth=1, markersize=3)
    # synthid auroc
    ax.plot([dp.temperature for dp in obf_dps], 
             [dp.auroc for dp in obf_dps], '#66aa33',
             marker="o", linewidth=1, markersize=3)
    
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax2.yaxis.set_major_locator(MultipleLocator(0.2))

    ax.set_ylim(0.45, 0.95)
    ax2.set_ylim(0, 1.05)

    ax.axhline(y=0.5, linestyle=(1.5, (3, 1.5)), 
               linewidth=2.5, color="pink")

    ax2.spines['left'].set_color('#ff3366')
    ax.set_ylabel(f'{dataset_name_map[dataset_name]} Pass@1', 
                  color='#ff3366')
    ax.tick_params(axis='y', colors='#ff3366') 

    ax2.spines['right'].set_color('purple')
    ax2.set_ylabel('AUROC', color='purple')
    ax2.tick_params(axis='y', colors='purple') 

    if add_x_label:
        ax.set_xlabel("Temperature")
    else:
        ax.tick_params(labelbottom=False, axis='x', 
                       which='both', length=0)
    ax.set_xticks([dp.temperature for dp in obf_dps])


if __name__ == "__main__":
    result_root = "data/result"
    figure_output_root = "data/figure"

    selected_models = ["Llama31Instruct8B"]
    lang_2_selected_obf = {
        "py": ["pyminify"],
        "js": ["javascript-obfuscator"]
    }
    dataset_names = ["humaneval_py", "humaneval_js", 
                     "mbpp_py", "mbpp_js"]
    plt.rcParams.update({
        'font.size': 8,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times"], 
    })

    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(4, 6))
    for model_name, lang in product(selected_models, lang_2_selected_obf.keys()):
        selected_obfuscators = lang_2_selected_obf[lang]
        def ff_name_2_row(ff_name):
            for row, d in enumerate(dataset_names):
                if d in ff_name:
                    return row, d

        for figure_folder_name in os.listdir(result_root):
            if "synthid" not in figure_folder_name:
                continue
            if model_name not in figure_folder_name:
                continue
            figure_root = f"{result_root}/{figure_folder_name}"
            dp_idxs = os.listdir(figure_root)
            dp_idxs = [i for i in dp_idxs if is_idx(i)]
            obf_2_dps: Dict[Tuple[str], List[DataPoint]] = {}
            for dp_idx in dp_idxs:
                metric_path = f"{figure_root}/{dp_idx}/metrics.jsonl"
                if os.path.exists(metric_path):
                    with open(metric_path) as file:
                        dps = [DataPoint(**json.loads(l)) for l in file]
                    for dp in dps:
                        if dp.obf_name not in selected_obfuscators and \
                                dp.obf_name != "Original":
                            continue
                        if dp.obf_name not in obf_2_dps:
                            obf_2_dps[dp.obf_name] = []
                        obf_2_dps[dp.obf_name].append(dp)
            ori_dps = obf_2_dps["Original"]
            del obf_2_dps["Original"]
            for obf_name, dps in obf_2_dps.items():
                ax_row, dataset_name = ff_name_2_row(figure_folder_name)
                ax = axs[ax_row]
                draw_plot(ax, dps, ori_dps, 
                          add_x_label=ax_row == len(dataset_names) - 1,
                          add_left_y_label=True,
                          add_right_y_label=True,
                          add_h_line_tag=True, 
                          model_name=model_name,
                          obf_name=obf_name,
                          dataset_name=dataset_name)
    fig.suptitle(model_name_map[model_name], fontsize=11)
    plt.tight_layout(h_pad=0.6)
    plt.savefig(f"{figure_output_root}/synthid_main--{model_name}.pdf")

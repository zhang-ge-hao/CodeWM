import os
import json
import sys
from typing import Dict, List, Tuple, Any
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
              add_h_line_tag, add_legend, add_title, model_name, 
              obf_name, dataset_name, lang):
    ori_dps.sort(key=lambda dp: dp.temperature)
    obf_dps.sort(key=lambda dp: dp.temperature)
    assert len(ori_dps) == len(obf_dps) and \
        all(dp1.temperature == dp2.temperature \
            for dp1, dp2 in zip(ori_dps, obf_dps))

    left_axis_color = "#fca326"
    ori_auroc_color = "#fca326"
    obf_auroc_color = "#e24329"
    right_axis_color = "#0087b4"

    # add right Y
    ax2 = ax.twinx()
    # pass1
    ax2.plot([dp.temperature for dp in obf_dps], 
            [dp.pass1 for dp in obf_dps], right_axis_color,
            marker="o", linewidth=1, markersize=4, 
            label="Pass@1")
    # ori auroc
    ax.plot([dp.temperature for dp in obf_dps], 
             [dp.auroc for dp in ori_dps], ori_auroc_color,
             marker="o", linewidth=1, markersize=4, 
             label="Original WMed")
    # synthid auroc
    ax.plot([dp.temperature for dp in obf_dps], 
             [dp.auroc for dp in obf_dps], obf_auroc_color,
             marker="o", linewidth=1, markersize=4, 
             label=f"Obfuscated")
    
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax2.yaxis.set_major_locator(MultipleLocator(0.2))

    ax.set_ylim(0.45, 0.95)
    ax2.set_ylim(0, 1.05)

    ax.axhline(y=0.5, **ref_line_config, 
               color=left_axis_color)
    ax2.spines['left'].set_color(left_axis_color)
    ax.set_ylabel(f'{dataset_name_map[dataset_name]}\nAUROC', 
                  color=left_axis_color)
    ax.tick_params(axis='y', colors=left_axis_color) 

    ax2.spines['right'].set_color(right_axis_color)
    ax2.set_ylabel('Pass@1', color=right_axis_color)
    ax2.tick_params(axis='y', colors=right_axis_color) 

    # legend
    if add_legend:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    if add_x_label:
        ax.set_xlabel("Temperature")
    else:
        ax.tick_params(labelbottom=False, axis='x', 
                       which='both', length=0)
    ax.set_xticks([dp.temperature for dp in obf_dps])
    if add_title:
        ax.set_title(f"Language: {lang_map[lang]}\nObfuscator: {obf_name_map[obf_name]}", 
                     loc="left")


def draw_line_chart(axs: List[Axes], dps: List[DataPoint], 
                    short_model_name, selected_obfuscators):
    selected_obfuscators = ["Original"] + selected_obfuscators
    pass1_lines: Dict[Any, List[DataPoint]] = {}
    auroc_lines: Dict[Any, List[DataPoint]] = {}

    def is_model_match(dp: DataPoint):
        _mn = dp.model_name.split("/")[1].replace("-", " ")
        return _mn.lower() == model_name_map[model_name].lower()

    __dsn_2_color = {}
    def dsn_2_color(dsn):
        if dsn not in __dsn_2_color:
            __dsn_2_color[dsn] = classical_set[len(__dsn_2_color)]
        return __dsn_2_color[dsn]

    for dp in dps:
        if not is_model_match(dp):
            continue
        if dp.watermarking == "no_wm" or \
                (dp.watermarking == "synthid" and dp.obf_name == "Original"):
            data_dict = pass1_lines
            key = (dp.watermarking, dp.dataset_name)
            if key not in data_dict:
                data_dict[key] = []
            data_dict[key].append(dp)

        if dp.watermarking == "synthid" and dp.obf_name in selected_obfuscators:
            data_dict = auroc_lines
            key = (dp.dataset_name, dp.obf_name)
            if key not in data_dict:
                data_dict[key] = []
            data_dict[key].append(dp)
    
    pass1_ax = axs[0]
    auroc_ax = axs[1]

    for (wm, ds), line_dps in pass1_lines.items():
        line_dps.sort(key=lambda dp: dp.temperature)
        ds_name = line_dps[0].dataset_name
        wm = line_dps[0].watermarking
        pass1_ax.plot([dp.temperature for dp in line_dps], 
                      [dp.pass1 for dp in line_dps], dsn_2_color(ds_name),
                      marker="o" if wm == "no_wm" else "s", 
                      linewidth=1, markersize=4, 
                      label=ds_name)
    
    for (ds, obf), line_dps in auroc_lines.items():
        line_dps.sort(key=lambda dp: dp.temperature)
        ds_name = line_dps[0].dataset_name
        obf = line_dps[0].obf_name
        auroc_ax.plot([dp.temperature for dp in line_dps], 
                      [dp.auroc for dp in line_dps], dsn_2_color(ds_name),
                      marker="o" if obf == "Original" else "s", 
                      linewidth=1, markersize=4, 
                      label=ds_name)
    pass1_ax.legend()
    auroc_ax.legend()


if __name__ == "__main__":
    result_root = "data/result"
    figure_output_root = "data/figure"

    selected_models = ["Llama31Instruct8B", "DSCoderBase33B"]
    cn_and_selected_obfs = [
        ("good_obf", ["pyminify", "javascript-obfuscator"]),
        ("all", ["pyminify", "pyminifier", 
                 "javascript-obfuscator", "uglifyjs"])
    ]
    plt.rcParams.update({
        'font.size': 8,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times"], 
    })

    dps = []
    for figure_folder_name in os.listdir(result_root):
        figure_root = f"{result_root}/{figure_folder_name}"
        dp_idxs = os.listdir(figure_root)
        dp_idxs = [i for i in dp_idxs if is_idx(i)]
        for dp_idx in dp_idxs:
            metric_path = f"{figure_root}/{dp_idx}/metrics.jsonl"
            if os.path.exists(metric_path):
                with open(metric_path) as file:
                    dps.extend([DataPoint(**json.loads(l)) for l in file])

    for model_name, (config_name, selected_obfuscators) in \
            product(selected_models, cn_and_selected_obfs):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
        draw_line_chart(axs, dps, model_name, selected_obfuscators)
        fig.suptitle(model_name_map[model_name], fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{figure_output_root}/synthid--{model_name}--{config_name}.pdf")

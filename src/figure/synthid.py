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


def draw_line_chart(fig, axs: List[Axes], dps: List[DataPoint], 
                    short_model_name, selected_obfuscators):
    selected_obfuscators = ["Original"] + selected_obfuscators
    pass1_lines: Dict[Any, List[DataPoint]] = {}
    auroc_lines: Dict[Any, List[DataPoint]] = {}

    def is_model_match(dp: DataPoint):
        _mn = dp.model_name.split("/")[1].replace("-", " ")
        return _mn.lower() == model_name_map[model_name].lower()

    __obf_2_marker = {}
    def obf_2_marker(obf):
        if obf not in __obf_2_marker:
            __obf_2_marker[obf] = marker_set[len(__obf_2_marker)]
        return __obf_2_marker[obf]

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
                    #   marker="o" if wm == "no_wm" else "s", 
                      linewidth=2, markersize=4, 
                      linestyle="-" if wm == "no_wm" else "--", 
                      label=ds_name)
    
    for (ds, obf), line_dps in auroc_lines.items():
        line_dps.sort(key=lambda dp: dp.temperature)
        ds_name = line_dps[0].dataset_name
        obf = line_dps[0].obf_name
        auroc_ax.plot([dp.temperature for dp in line_dps], 
                    [dp.auroc for dp in line_dps], 
                    color=dsn_2_color(ds_name),
                    marker=obf_2_marker(obf), 
                    markerfacecolor="white",
                    markeredgecolor=dsn_2_color(ds_name),
                    markeredgewidth=1.5,
                    linewidth=2, markersize=6, 
                    label=ds_name)

    pass1_ax.set_ylabel("Pass@1")
    auroc_ax.set_ylabel("AUROC")
    pass1_ax.set_xlabel("Temperature")
    auroc_ax.set_xlabel("Temperature")
    pass1_ax.set_title("Code Generation Performance\nBef./Aft. Watermarking")
    auroc_ax.set_title("Watermarking Detection Performance\nBef./Aft. Obfuscation")

    pass1_ax.set_xticks([dp.temperature for dp in dps])
    auroc_ax.set_xticks([dp.temperature for dp in dps])
    # pass1_ax.legend()
    # auroc_ax.legend()

    line_2ds = []
    datasets = ["humaneval_py", "humaneval_js", "mbpp_py", "mbpp_js"]
    for dsn in datasets:
        color = __dsn_2_color[dsn]
        line_2ds.append(
            Line2D([0], [0], 
                   marker="s", 
                   label=dataset_name_map[dsn],
                   markerfacecolor=color, 
                   markeredgecolor="none",
                   markersize=7,
                   linestyle='none')
        )
    legend = fig.legend(
        handles=line_2ds, ncol=1, title="Dataset (Line Color)       ",
        loc='upper left', title_fontsize=7, 
        fontsize=7, frameon=True,
        bbox_to_anchor=(0.82, 0.89))
    # auroc_ax.add_artist(legend)
    legend.get_title().set_ha('left')
    legend._legend_box.align = "left"

    line_2ds = []

    obf_names = ["Original", "pyminify", "pyminifier", 
                 "javascript-obfuscator", "uglifyjs"]
    for obf_name in obf_names:
        line_2ds.append(
            Line2D([0], [0], 
                   marker=obf_2_marker(obf_name), 
                   label=obf_name_map.get(obf_name, "Original"),
                   markerfacecolor="white", 
                   markeredgecolor="darkgrey",
                   markersize=7,
                   markeredgewidth=1.5,
                   linestyle='none')
        )
    legend = fig.legend(
        handles=line_2ds, ncol=1, title="Obfuscation (Marker)     ",
        loc='upper left', title_fontsize=7, 
        fontsize=7, frameon=True,
        bbox_to_anchor=(0.82, 0.63))
    # auroc_ax.add_artist(legend)
    legend.get_title().set_ha('left')
    legend._legend_box.align = "left"

    line_2ds = []
    for label, linestyle in [("Non-WMed", "-"), ("SynthID WMed", "--")]:
        line_2ds.append(
            Line2D([0], [0], 
                   label=label,
                   color="darkgrey",
                   marker="none",
                   linewidth=2,
                   linestyle=linestyle)
        )
    legend = fig.legend(
        handles=line_2ds, ncol=1, title="Bef./Aft. WM (Line Style)",
        loc='lower left', title_fontsize=7, 
        fontsize=7, frameon=True,
        bbox_to_anchor=(0.82, 0.14))
    # auroc_ax.add_artist(legend)
    legend.get_title().set_ha('left')
    legend._legend_box.align = "left"

if __name__ == "__main__":
    result_root = "data/result"
    figure_output_root = "data/figure"

    selected_models = ["Llama31Instruct8B", "DSCoderBase33B"]
    cn_and_selected_obfs = [
        # ("good_obf", ["pyminify", "javascript-obfuscator"]),
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
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 3.1))
        draw_line_chart(fig, axs, dps, model_name, selected_obfuscators)
        # fig.suptitle(
        #     f"{model_name_map[model_name]} vs. SynthID", fontsize=11
        # )
        plt.tight_layout()
        fig_path = f"{figure_output_root}/synthid--{model_name}--{config_name}.pdf"
        if os.path.exists(fig_path):
            os.remove(fig_path)
        fig.subplots_adjust(right=0.81, wspace=0.22)
        plt.savefig(fig_path)

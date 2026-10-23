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
              add_title, add_x_label, add_y_label, add_h_line_tag, 
              dataset_name, threshold_ratio=0.8):
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
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(True, which='major', **major_gird_config,
            zorder=zo())

    # Points
    for dp in dps:
        ax.scatter(
            x=dp.pass1, 
            y=dp.auroc, 
            color=get_color(dp), 
            marker=get_marker(dp),
            edgecolors=get_edgecolor(dp), 
            s=40,
            linewidths=1.2,
            zorder=zo())
    
    # Reference line
    ax.axvline(x=no_wm_res * threshold_ratio, **ref_line_config, 
               color=v_line_color, zorder=zo())
    ax.axhline(y=0.5, **ref_line_config, 
               color=h_line_color, zorder=zo())
    
    # Range
    ax.set_ylim(0.2, 1.05)

    # Reference line Tag
    bbox = dict(facecolor='white',
                alpha=0.75,
                edgecolor='none',
                boxstyle='round,pad=0.0')
    tag_config = {
        "zorder": zo()
    }
    x_min, x_max = ax.get_xlim()
    v_tag = f"Pass@1 \u25BC {100 - int(threshold_ratio * 100)}%"
    ha = "right" if x_min < 0.5 else "left"
    ax.text(no_wm_res * threshold_ratio, 0.21, v_tag, 
            ha=ha, va='bottom', color=v_line_color,
            **tag_config)
    label_x_ratio = 1 / 15
    tag_box = None
    for dp in dps:
        if dp.auroc < 0.5 and \
                dp.pass1 < x_min + label_x_ratio * (x_max - x_min):
            tag_box = bbox
    if add_h_line_tag:
        ax.text(x_min, 0.49, "AUROC = 0.50", ha="left", va="top", 
                color=h_line_color, 
                **tag_config, bbox=tag_box)

    # Title and Label
    if add_x_label:
        ax.set_xlabel(f"Pass@1\n{dataset_name}")

    if dps[0].obf_name == "Original":
        title = "Original WM"
    else:
        title = obf_name_map[dps[0].obf_name] + " Obfuscated"
    if add_title:
        ax.set_title(title)

    if add_x_label:
        ax.set_xlabel(f"Pass@1")

    if add_y_label:
        ax.set_ylabel(f"{dataset_name}\nAUROC")
    else:
        ax.tick_params(labelleft=False)


if __name__ == "__main__":
    result_root = "data/result"
    figure_output_root = "data/figure"

    selected_models = ["Llama31Instruct8B", "DSCoderBase33B"]
    lang_2_selected_obf = {
        "py": ["pyminify"],
        "js": ["javascript-obfuscator"]
    }
    lang_2_ds_ordered = {
        "py": ["humaneval_py", "mbpp_py"],
        "js": ["humaneval_js", "mbpp_js"],
    }
    plt.rcParams.update({
        'font.size': 8,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times"], 
    })

    for model_name, lang in product(selected_models, lang_2_selected_obf.keys()):
        selected_obfuscators = lang_2_selected_obf[lang]
        selected_obfuscators.append("Original")
        dataset_ordered = lang_2_ds_ordered[lang]
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 5),
                                sharey="row")
        def ff_name_2_row(ff_name):
            for r_id, d in enumerate(dataset_ordered):
                if d in ff_name:
                    return r_id
        def obf_name_2_col(obf_name):
            if obf_name == "Original":
                return 0
            else:
                return 1

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
                if dp_demo.obf_name not in selected_obfuscators:
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
                          add_title=ax_row == 0,
                          add_x_label=ax_row == len(dataset_ordered) - 1,
                          add_y_label=ax_col == 0, 
                          add_h_line_tag=ax_col == 0, 
                          dataset_name=dp_demo.dataset_name)
                if ax_row == 0 and ax_col == 1:
                    add_gamma_legend(ax)
                    add_delta_legend(ax)
                    add_entropy_threshold_legend(ax)
        # fig.suptitle(f"{model_name_map[model_name]} vs. {lang_map[lang]}", 
        #              fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{figure_output_root}/g_r_main--{model_name}--{lang}.pdf")

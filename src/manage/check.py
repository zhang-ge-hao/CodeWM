import os
import sys
import json
import sys
from dataclasses import asdict
sys.path.append("src")

from _data import *
from metrics import calculate

def all_exps():
    for fn1 in os.listdir("data/result"):
        for fn2 in os.listdir(f"data/result/{fn1}"):
            gp = f"data/result/{fn1}/{fn2}/generate.jsonl"
            op = f"data/result/{fn1}/{fn2}/obfuscate.jsonl"
            mp = f"data/result/{fn1}/{fn2}/metrics.jsonl"
            if "backup" in mp:
                continue
            yield fn1, fn2, gp, op, mp


action = sys.argv[1]

if action == "finish":
    benchmark_map = {
        "humaneval_py": 164,
        "humaneval_js": 164,
        "mbpp_py": 378,
        "mbpp_js": 397,
    }
    for fn1, fn2, gp, op, mp in all_exps():
        unfinished = False
        if any(not os.path.exists(p) for p in [gp, op, mp]):
            unfinished = True
        if not unfinished:
            with open(gp) as file:
                g_ed_c = len(list(file))
            for bn, exp_c in benchmark_map.items():
                if bn in fn1 and g_ed_c != exp_c:
                    unfinished = True
        if unfinished:
            print(fn1, fn2)
if action == "bad_trans":
    all_bad_trans = {}
    tot = {}
    single_tuple_count = 0
    for fn1, fn2, gp, op, mp in all_exps():
        with open(gp) as file:
            gen_tasks = [GenTask(**json.loads(l)) for l in file]
            id_2_gen_task = {t.id: t for t in gen_tasks}
        with open(op) as file:
            for line in file:
                obf_task = ObfTask(**json.loads(line))
                if obf_task.obf_name not in tot:
                    all_bad_trans[obf_task.obf_name] = 0
                    tot[obf_task.obf_name] = 0
                gen_task = id_2_gen_task[obf_task.gen_task_id]
                if obf_task.bad_trans and gen_task.passed:
                    all_bad_trans[obf_task.obf_name] += 1
                    print(gen_task.solution)
                    is_single_tuple = ",)" in gen_task.solution.replace(" ", "")
                    single_tuple_count += 1 if is_single_tuple else 0
                tot[obf_task.obf_name] += 1
    print(json.dumps(all_bad_trans, indent=4))
    print(json.dumps(tot, indent=4))
    print(single_tuple_count)
if action == "metrics":
    for fn1, fn2, gp, op, mp in all_exps():
        with open(gp) as file:
            gen_tasks = [GenTask(**json.loads(l)) for l in file]
        with open(op) as file:
            obf_tasks = [ObfTask(**json.loads(l)) for l in file]
        dps = calculate(gen_tasks, obf_tasks)
        with open(mp, "w") as file:
            for dp in dps:
                file.write(json.dumps(asdict(dp)) + "\n")
if action == "synthid_diff":
    pairs = {}
    temp_2_auroc = {}
    for fn1, fn2, gp, op, mp in all_exps():
        with open(mp) as file:
            dps = [DataPoint(**json.loads(l)) for l in file]
        for dp in dps:
            if dp.watermarking == "no_wm" or \
                    (dp.watermarking == "synthid" and dp.obf_name == "Original"):
                key = (dp.model_name, dp.dataset_name, dp.temperature)
                if key not in pairs:
                    pairs[key] = []
                if dp.watermarking == "no_wm":
                    pairs[key] = [dp.pass1] + pairs[key]
                else:
                    pairs[key] = pairs[key] + [dp.pass1]
                if dp.watermarking == "synthid":
                    if dp.temperature not in temp_2_auroc:
                        temp_2_auroc[dp.temperature] = []
                    temp_2_auroc[dp.temperature].append(dp.auroc)
    for temp, auroc_list in temp_2_auroc.items():
        print(f"temp: {temp}")
        print(f"auroc: {sum(auroc_list) / len(auroc_list)}")
    __pairs = {}
    for (mn, ds, _), p in pairs.items():
        if (mn, ds) not in __pairs:
            __pairs[(mn, ds)] = []
        __pairs[(mn, ds)].append(p)
    pairs = __pairs
    diff_sum = 0
    count = 0
    for (mn, ds), ps in pairs.items():
        diffs = [p2 - p1 for p1, p2 in ps]
        print(mn)
        print(ds)
        print(len([d for d in diffs if d >= 0]), len([d for d in diffs if d < 0]))
        # diffs.sort()
        # print(diffs)
        diff_sum += sum([d for d in diffs])
        count += len(diffs)
        print(sum([d for d in diffs]) / len(diffs))
        print()
    print(diff_sum / count)
if action == "comp_dist":
    dist = [0] * 11
    tot_comp_count = 0
    tot_tot_count = 0
    for fn1, fn2, gp, op, mp in all_exps():
        if "no_wm" in fn1:
            continue
        with open(gp) as file:
            gen_tasks = [GenTask(**json.loads(l)) for l in file]
        with open(op) as file:
            obf_tasks = [ObfTask(**json.loads(l)) for l in file]
        tot_count = len(gen_tasks)
        comp_set = set()
        for obf_task in obf_tasks:
            if obf_task.solution is not None:
                comp_set.add(obf_task.gen_task_id)
        ratio = len(comp_set) / tot_count
        dist[int(ratio * 10)] += 1
        tot_comp_count += len(comp_set)
        tot_tot_count += tot_count
    print(dist)
    print(sum(dist[-2: ]) / sum(dist))
    print(sum(dist[-3: ]) / sum(dist))
    print(sum(dist))
    print(tot_comp_count, tot_tot_count)
    print(tot_comp_count / tot_tot_count)
if action == "length":
    import numpy as np
    token_counts = []
    code_count = 0
    for fn1, fn2, gp, op, mp in all_exps():
        if "no_wm" in fn1:
            continue
        with open(gp) as file:
            gen_tasks = [GenTask(**json.loads(l)) for l in file]
        for gen_task in gen_tasks:
            token_counts.append(gen_task.s_len)
        code_count += len(gen_tasks)
    print(sum(token_counts))
    print(code_count)
    print(sum(token_counts) / code_count)
    percentile_90 = np.percentile(token_counts, 90)
    print(percentile_90)
if action == "centralize":
    import numpy as np
    from scipy import stats
    obf_2_roc = {}
    good_obfs = ["pyminify", "javascript-obfuscator"]
    for fn1, fn2, gp, op, mp in all_exps():
        if "no_wm" in fn1:
            continue
        with open(mp) as file:
            dps = [DataPoint(**json.loads(l)) for l in file]
        # dps = [dp for dp in dps if dp.obf_name in good_obfs]
        for dp in dps:
            if dp.obf_name == "Original":
                continue
            if dp.obf_name not in obf_2_roc:
                obf_2_roc[dp.obf_name] = []
            obf_2_roc[dp.obf_name].append(dp.auroc)
    for obf_name, rocs in obf_2_roc.items():
        print(obf_name)
        print(f"mean roc: {sum(rocs) / len(rocs)}")
        for r in [1, 5, 10]:
            r /= 100
            count = sum(
                [1 if 0.5 - r < roc < 0.5 + r else 0 for roc in rocs])
            print(count)
            print(f"diff: {r:.3f}; not in: {len(rocs)-count}; ratio: {count / len(rocs):.3f}")
        print(f"exp_count: {len(rocs)}")

        mean = np.mean(rocs)
        std_dev = np.std(rocs, ddof=1)
        se = std_dev / np.sqrt(len(rocs))

        confidence = 0.95
        df = len(rocs) - 1
        t_critical = stats.t.ppf((1 + confidence) / 2, df)

        margin_of_error = t_critical * se
        print(f"std: {std_dev:.3f}, error: {margin_of_error:.6f}")
if action == "maintain":
    no_wm_temp1_model_n_ds_2_pass1 = {}
    for fn1, fn2, gp, op, mp in all_exps():
        if "no_wm" in fn1:
            with open(mp) as file:
                dps = [DataPoint(**json.loads(l)) for l in file]
            dp = dps[0]
            if dp.temperature == 1.0:
                no_wm_temp1_model_n_ds_2_pass1[(
                    dp.model_name, dp.dataset_name)] = dp.pass1
        # dps = [dp for dp in dps if dp.obf_name in good_obfs]
    maintained_dps = []
    total_count = 0
    for fn1, fn2, gp, op, mp in all_exps():
        if "no_wm" in fn1:
            continue
        with open(mp) as file:
            dps = [DataPoint(**json.loads(l)) for l in file]
        for dp in dps:
            if dp.obf_name == "Original":
                continue
            total_count += 1
            if dp.pass1 > 0.8 * no_wm_temp1_model_n_ds_2_pass1[(
                    dp.model_name, dp.dataset_name)] and dp.auroc > 0.55:
                maintained_dps.append(dp)
    maintained_dps.sort(key=lambda dp: -dp.auroc)
    for dp in maintained_dps:
        print(dp)
    print(len(maintained_dps))
    print(total_count)
if action == "update_n_gram":
    for fn1, fn2, gp, op, mp in all_exps():
        if "no_wm" in fn1:
            ngram_len = None
        else:
            ngram_len = 5
        with open(mp) as file:
            dp_dicts = [json.loads(l) for l in file]
        
        with open(mp, "w") as file:
            for dp_dict in dp_dicts:
                dp_dict["ngram_len"] = ngram_len
                file.write(json.dumps(dp_dict) + "\n")
        
        with open(gp) as file:
            gen_t_dicts = [json.loads(l) for l in file]
        with open(gp, "w") as file:
            for gen_task in gen_t_dicts:
                gen_task["ngram_len"] = ngram_len
                file.write(json.dumps(gen_task) + "\n")
if action == "tmp_cpp":
    import numpy as np
    ori_auroc_list = []
    obf_auroc_list = []
    for fn1, fn2, gp, op, mp in all_exps():
        if "humaneval_cpp" in fn1 and "wllm" in fn1:
            with open(mp) as file:
                dps = [DataPoint(**json.loads(l)) for l in file]
            for dp in dps:
                if dp.obf_name == "Original":
                    ori_auroc_list.append(dp.auroc)
                elif dp.obf_name == "stunnix":
                    obf_auroc_list.append(dp.auroc)
    print(len(ori_auroc_list))
    print(len(obf_auroc_list))
    ori_mean = np.mean(ori_auroc_list)
    obf_mean = np.mean(obf_auroc_list)
    obf_std = np.std(obf_auroc_list, ddof=1)

    print(f"{ori_mean} -> {obf_mean}")
    print(obf_std)
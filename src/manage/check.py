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
    for fn1, fn2, gp, op, mp in all_exps():
        with open(op) as file:
            for line in file:
                obf_name = json.loads(line)["obf_name"]
                bad_trans = json.loads(line)["bad_trans"]
                if bad_trans:
                    if obf_name not in all_bad_trans:
                        all_bad_trans[obf_name] = 0
                    all_bad_trans[obf_name] += 1
    print(json.dumps(all_bad_trans, indent=4))
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
    __pairs = {}
    for (mn, ds, _), p in pairs.items():
        if (mn, ds) not in __pairs:
            __pairs[(mn, ds)] = []
        __pairs[(mn, ds)].append(p)
    pairs = __pairs
    for (mn, ds), ps in pairs.items():
        diffs = [p2 - p1 for p1, p2 in ps]
        print(mn)
        print(ds)
        print(len([d for d in diffs if d >= 0]), len([d for d in diffs if d < 0]))
        # diffs.sort()
        # print(diffs)
        print(sum([d for d in diffs]) / len(diffs))
        print()

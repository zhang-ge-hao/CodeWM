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
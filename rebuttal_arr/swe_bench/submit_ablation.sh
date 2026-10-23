#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

phase="${1:-}"
if [[ "${phase}" != "smoke" && "${phase}" != "remaining" ]]; then
    echo "Usage: $0 {smoke|remaining}" >&2
    exit 2
fi

root="data/ablation"
mkdir -p "${root}"

submit_non_wm() {
    local name="non-wm"
    local output="${root}/${name}"
    mkdir -p "${output}"
    sbatch --parsable \
        --job-name="sweb-${name}" \
        --output="${output}/slurm-%j.out" \
        --error="${output}/slurm-%j.err" \
        --export="ALL,WATERMARKING=none,EXPERIMENT_DATA_DIR=${output},EXPERIMENT_RUN_ID=${name}" \
        run_ablation.sbatch
}

submit_wllm() {
    local delta="$1"
    local gamma="$2"
    local name="wllm-delta-${delta}-gamma-${gamma}-ngram-5"
    local output="${root}/${name}"
    mkdir -p "${output}"
    sbatch --parsable \
        --job-name="sweb-d${delta}-g${gamma}" \
        --output="${output}/slurm-%j.out" \
        --error="${output}/slurm-%j.err" \
        --export="ALL,WATERMARKING=wllm,DELTA=${delta},GAMMA=${gamma},EXPERIMENT_DATA_DIR=${output},EXPERIMENT_RUN_ID=${name}" \
        run_ablation.sbatch
}

if [[ "${phase}" == "smoke" ]]; then
    submit_non_wm
    submit_wllm 0.5 0.1
    submit_wllm 4 0.5
    exit 0
fi

for delta in 0.5 1 2 3 4; do
    for gamma in 0.1 0.25 0.5; do
        if [[ "${delta}/${gamma}" == "0.5/0.1" || "${delta}/${gamma}" == "4/0.5" ]]; then
            continue
        fi
        submit_wllm "${delta}" "${gamma}"
    done
done

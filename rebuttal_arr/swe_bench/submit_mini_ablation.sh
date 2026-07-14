#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

phase="${1:-}"
if [[ "${phase}" != "smoke" && "${phase}" != "smoke-retry" && "${phase}" != "gate" && "${phase}" != "full" ]]; then
    echo "Usage: $0 {smoke|smoke-retry|gate|full}" >&2
    exit 2
fi

selection_source="${SELECTION_SOURCE:-data/selection.json}"
test -f "${selection_source}"

submit_non_wm() {
    local stage="$1"
    local count="$2"
    local name="non-wm"
    local output="data/ablation-mini-swe-agent/${stage}/${name}"
    mkdir -p "${output}"
    sbatch --parsable \
        --job-name="mini-${stage}-nonwm" \
        --output="${output}/slurm-%j.out" \
        --error="${output}/slurm-%j.err" \
        --export="ALL,WATERMARKING=none,SAMPLE_SIZE=${count},SELECTION_SOURCE=${selection_source},EXPERIMENT_DATA_DIR=${output},EXPERIMENT_RUN_ID=mini-${stage}-${name}" \
        run_mini_ablation.sbatch
}

submit_wllm() {
    local stage="$1"
    local count="$2"
    local delta="$3"
    local gamma="$4"
    local name="wllm-delta-${delta}-gamma-${gamma}-ngram-5"
    local output="data/ablation-mini-swe-agent/${stage}/${name}"
    mkdir -p "${output}"
    sbatch --parsable \
        --job-name="mini-${stage}-d${delta}-g${gamma}" \
        --output="${output}/slurm-%j.out" \
        --error="${output}/slurm-%j.err" \
        --export="ALL,WATERMARKING=wllm,DELTA=${delta},GAMMA=${gamma},SAMPLE_SIZE=${count},SELECTION_SOURCE=${selection_source},EXPERIMENT_DATA_DIR=${output},EXPERIMENT_RUN_ID=mini-${stage}-${name}" \
        run_mini_ablation.sbatch
}

if [[ "${phase}" == "smoke" ]]; then
    selection_source="data/ablation-mini-swe-agent/smoke-selection.json"
    submit_non_wm smoke-3 3
    exit 0
fi

if [[ "${phase}" == "smoke-retry" ]]; then
    selection_source="data/ablation-mini-swe-agent/smoke-retry-selection.json"
    submit_non_wm smoke-2-sequential 2
    exit 0
fi

if [[ "${phase}" == "gate" ]]; then
    submit_non_wm gate-10-timecap300 10
    submit_wllm gate-10-timecap300 10 0.5 0.1
    submit_wllm gate-10-timecap300 10 4 0.5
    exit 0
fi

submit_non_wm full-50 50
for delta in 0.5 1 2 3 4; do
    for gamma in 0.1 0.25 0.5; do
        submit_wllm full-50 50 "${delta}" "${gamma}"
    done
done

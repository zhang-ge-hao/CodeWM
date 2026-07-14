#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

phase="${1:-}"
if [[ "${phase}" != "gate" && "${phase}" != "remaining" ]]; then
    echo "Usage: $0 {gate|remaining}" >&2
    exit 2
fi

selection_source="${SELECTION_SOURCE:-data/selection.json}"
root="${DATA_ROOT:-data/ablation-mini-swe-agent-vllm/qwen35-gptq-int4/full-50}"
test -f "${selection_source}"
mkdir -p "${root}"

submit_one() {
    local name="$1"
    local watermarking="$2"
    local delta="${3:-}"
    local gamma="${4:-}"
    local output="${root}/${name}"
    local marker="${output}/.submitted"
    mkdir -p "${output}"
    if [[ -s "${marker}" ]]; then
        echo "already submitted: ${name} job=$(cat "${marker}")" >&2
        return
    fi

    local exports
    exports="ALL,WATERMARKING=${watermarking},SAMPLE_SIZE=50,SELECTION_SOURCE=${selection_source},EXPERIMENT_DATA_DIR=${output},EXPERIMENT_RUN_ID=q35gptq-${name},AGENT_WORKERS=25,AGENT_WALL_TIME_LIMIT_SECONDS=1800,MAX_NEW_TOKENS=10000"
    if [[ "${watermarking}" == "wllm" ]]; then
        exports+=",DELTA=${delta},GAMMA=${gamma}"
    fi

    local job_id
    job_id="$(sbatch --parsable \
        --job-name="q35-${name}" \
        --output="${output}/slurm-%j.out" \
        --error="${output}/slurm-%j.err" \
        --export="${exports}" \
        run_mini_vllm_ablation.sbatch)"
    printf '%s\n' "${job_id}" >"${marker}"
    printf '%s\t%s\n' "${job_id}" "${name}"
}

submit_gate() {
    submit_one non-wm none
    submit_one wllm-delta-0.5-gamma-0.1-ngram-5 wllm 0.5 0.1
    submit_one wllm-delta-4-gamma-0.5-ngram-5 wllm 4 0.5
}

submit_remaining() {
    # Keep the three Slurm jobs close in expected wall time. Larger delta is
    # treated as more likely to damage agent behavior and create long cases.
    submit_bundle bundle-a "4:0.25;3:0.1;2:0.5;1:0.25"
    submit_bundle bundle-b "4:0.1;3:0.25;2:0.25;1:0.5"
    submit_bundle bundle-c "3:0.5;2:0.1;1:0.1;0.5:0.5;0.5:0.25"
}

submit_bundle() {
    local bundle_name="$1"
    local configs="$2"
    local bundle_dir="${root}/bundles/${bundle_name}"
    local marker="${bundle_dir}/.submitted"
    mkdir -p "${bundle_dir}"
    if [[ -s "${marker}" ]]; then
        echo "already submitted: ${bundle_name} job=$(cat "${marker}")" >&2
        return
    fi

    local item delta gamma name output
    IFS=';' read -r -a items <<<"${configs}"
    for item in "${items[@]}"; do
        IFS=':' read -r delta gamma <<<"${item}"
        name="wllm-delta-${delta}-gamma-${gamma}-ngram-5"
        output="${root}/${name}"
        if [[ -s "${output}/.submitted" ]]; then
            echo "refusing partial bundle: ${name} is already submitted" >&2
            return 1
        fi
    done

    local exports
    exports="ALL,BUNDLE_CONFIGS=${configs},BUNDLE_OUTPUT_ROOT=${root},BUNDLE_JOB_DIR=${bundle_dir},SAMPLE_SIZE=50,SELECTION_SOURCE=${selection_source},AGENT_WORKERS=25,AGENT_WALL_TIME_LIMIT_SECONDS=1800,ENVIRONMENT_START_ATTEMPTS=5,MAX_NEW_TOKENS=10000"
    local job_id
    job_id="$(sbatch --parsable \
        --job-name="q35-${bundle_name}" \
        --output="${bundle_dir}/slurm-%j.out" \
        --error="${bundle_dir}/slurm-%j.err" \
        --export="${exports}" \
        run_mini_vllm_ablation.sbatch)"
    printf '%s\n' "${job_id}" >"${marker}"
    for item in "${items[@]}"; do
        IFS=':' read -r delta gamma <<<"${item}"
        name="wllm-delta-${delta}-gamma-${gamma}-ngram-5"
        output="${root}/${name}"
        mkdir -p "${output}"
        printf '%s\n' "${job_id}" >"${output}/.submitted"
    done
    printf '%s\t%s\t%s\n' "${job_id}" "${bundle_name}" "${configs}"
}

if [[ "${phase}" == "gate" ]]; then
    submit_gate
fi
if [[ "${phase}" == "remaining" ]]; then
    submit_remaining
fi

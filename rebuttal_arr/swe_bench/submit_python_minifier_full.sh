#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

source_root="${SOURCE_ROOT:-rebuttal_arr/swe_bench/data/ablation-mini-swe-agent-vllm/qwen35-gptq-int4/full-50}"
output_root="${OUTPUT_ROOT:-rebuttal_arr/swe_bench/data/ablation-mini-swe-agent-vllm/qwen35-gptq-int4/full-50-python-minifier-method/full-50}"
mkdir -p "${output_root}/bundles"

submit_bundle() {
    local bundle="$1"
    local names="$2"
    local bundle_dir="${output_root}/bundles/${bundle}"
    local marker="${bundle_dir}/.submitted"
    mkdir -p "${bundle_dir}"
    if [[ -s "${marker}" ]]; then
        printf 'already submitted: %s job=%s\n' "${bundle}" "$(cat "${marker}")" >&2
        return
    fi
    local job_id
    job_id="$(sbatch --parsable \
        --job-name="pymin-${bundle}" \
        --output="${bundle_dir}/slurm-%j.out" \
        --error="${bundle_dir}/slurm-%j.err" \
        --export="ALL,SOURCE_ROOT=${source_root},OUTPUT_ROOT=${output_root},BUNDLE_NAMES=${names},TRANSFORM_WORKERS=${TRANSFORM_WORKERS:-4},EVAL_WORKERS=4,PREPARE_ONLY=${PREPARE_ONLY:-0},SKIP_DETECTION=${SKIP_DETECTION:-0}" \
        rebuttal_arr/swe_bench/run_python_minifier_bundle.sbatch)"
    printf '%s\n' "${job_id}" >"${marker}"
    printf '%s\t%s\t%s\n' "${job_id}" "${bundle}" "${names}"
}

submit_bundle bundle-a \
    "non-wm:wllm-delta-4-gamma-0.1-ngram-5:wllm-delta-4-gamma-0.25-ngram-5:wllm-delta-4-gamma-0.5-ngram-5"
submit_bundle bundle-b \
    "wllm-delta-0.5-gamma-0.1-ngram-5:wllm-delta-0.5-gamma-0.25-ngram-5:wllm-delta-0.5-gamma-0.5-ngram-5:wllm-delta-3-gamma-0.5-ngram-5"
submit_bundle bundle-c \
    "wllm-delta-1-gamma-0.1-ngram-5:wllm-delta-1-gamma-0.25-ngram-5:wllm-delta-1-gamma-0.5-ngram-5:wllm-delta-3-gamma-0.25-ngram-5"
submit_bundle bundle-d \
    "wllm-delta-2-gamma-0.1-ngram-5:wllm-delta-2-gamma-0.25-ngram-5:wllm-delta-2-gamma-0.5-ngram-5:wllm-delta-3-gamma-0.1-ngram-5"

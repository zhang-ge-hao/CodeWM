#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${PI_WORKDIR:?PI_WORKDIR must point to the shared work directory}"

source_root="${SOURCE_ROOT:-rebuttal_arr/swe_bench/data/ablation-mini-swe-agent-vllm/qwen35-gptq-int4/full-50}"
output_root="${OUTPUT_ROOT:-rebuttal_arr/swe_bench/data/ablation-mini-swe-agent-vllm/qwen35-gptq-int4/full-50-python-minifier-method/timing-7-retained-solve}"
mkdir -p "${output_root}/bundles"

submit_bundle() {
    local bundle="$1"
    local names="$2"
    local bundle_dir="${output_root}/bundles/${bundle}"
    mkdir -p "${bundle_dir}"
    local job_id
    job_id="$(sbatch --parsable \
        --job-name="method-time-${bundle}" \
        --output="${bundle_dir}/slurm-%j.out" \
        --error="${bundle_dir}/slurm-%j.err" \
        --export="ALL,PI_WORKDIR=${PI_WORKDIR},SOURCE_ROOT=${source_root},OUTPUT_ROOT=${output_root},BUNDLE_NAMES=${names},TRANSFORM_WORKERS=${TRANSFORM_WORKERS:-8},PREPARE_ONLY=1,SKIP_DETECTION=1" \
        rebuttal_arr/swe_bench/run_python_minifier_bundle.sbatch)"
    printf '%s\t%s\t%s\n' "${job_id}" "${bundle}" "${names}"
}

submit_bundle bundle-a \
    "wllm-delta-2-gamma-0.1-ngram-5:wllm-delta-0.5-gamma-0.25-ngram-5"
submit_bundle bundle-b \
    "wllm-delta-1-gamma-0.25-ngram-5:wllm-delta-2-gamma-0.25-ngram-5"
submit_bundle bundle-c \
    "wllm-delta-0.5-gamma-0.5-ngram-5:wllm-delta-1-gamma-0.5-ngram-5"
submit_bundle bundle-d \
    "wllm-delta-2-gamma-0.5-ngram-5"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${PI_WORKDIR:?PI_WORKDIR must point to the shared work directory}"

diff_root="${DIFF_ROOT:-rebuttal_arr/swe_bench/data/ablation-mini-swe-agent-vllm/qwen35-gptq-int4/full-50-python-minifier-method/full-50-diff-only}"
method_root="${METHOD_ROOT:-rebuttal_arr/swe_bench/data/ablation-mini-swe-agent-vllm/qwen35-gptq-int4/full-50-python-minifier-method/full-50-z-only}"
if [[ ! -d "${method_root}" ]]; then
    cp -a "${diff_root}" "${method_root}"
fi
mkdir -p "${method_root}/detector-bundles"

submit_bundle() {
    local bundle="$1"
    local names="$2"
    local bundle_dir="${method_root}/detector-bundles/${bundle}"
    mkdir -p "${bundle_dir}"
    local job_id
    job_id="$(sbatch --parsable \
        --output="${bundle_dir}/slurm-%j.out" \
        --error="${bundle_dir}/slurm-%j.err" \
        --export="ALL,PI_WORKDIR=${PI_WORKDIR},METHOD_ROOT=${method_root},BUNDLE_NAMES=${names},DETECT_WORKERS=${DETECT_WORKERS:-8}" \
        rebuttal_arr/swe_bench/recompute_method_detection.sbatch)"
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

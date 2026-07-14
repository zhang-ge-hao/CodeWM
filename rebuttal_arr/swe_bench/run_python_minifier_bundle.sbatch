#!/usr/bin/env bash
#SBATCH --job-name=pymin-bundle
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=rebuttal_arr/swe_bench/data/ablation-mini-swe-agent-vllm/python-minifier-bundle-%j.out
#SBATCH --error=rebuttal_arr/swe_bench/data/ablation-mini-swe-agent-vllm/python-minifier-bundle-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?}"
: "${PI_WORKDIR:?PI_WORKDIR must point to the shared work directory}"
: "${SOURCE_ROOT:?Set the completed full-50 experiment root}"
: "${OUTPUT_ROOT:?Set the Python-Minifier full-50 output root}"
: "${BUNDLE_NAMES:?Set a colon-delimited list of experiment directory names}"

ENV_ROOT="${CONDA_ENV_ROOT:-${PI_WORKDIR}/conda/envs}"
QWEN_PYTHON="${ENV_ROOT}/qwen36/bin/python"
SWEBENCH_PYTHON="${ENV_ROOT}/swebench/bin/python"
PYMINIFY="${PYMINIFY_BIN:-${HOME}/conda/envs/watermarking/bin/pyminify}"
REPO_CACHE="${REPO_CACHE:-rebuttal_arr/swe_bench/data/ablation-mini-swe-agent-vllm/repo-cache}"

for executable in "${QWEN_PYTHON}" "${SWEBENCH_PYTHON}" "${PYMINIFY}"; do
    test -x "${executable}"
done
mkdir -p "${OUTPUT_ROOT}" "${REPO_CACHE}"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${PI_WORKDIR}/huggingface}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export MODAL_BUILD_VALIDATION=ignore
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="${PWD}:${PWD}/rebuttal_arr/swe_bench${PYTHONPATH:+:${PYTHONPATH}}"

IFS=':' read -r -a names <<<"${BUNDLE_NAMES}"
for name in "${names[@]}"; do
    source_dir="${SOURCE_ROOT}/${name}"
    output_dir="${OUTPUT_ROOT}/${name}"
    test -f "${source_dir}/summary.json"
    mkdir -p "${output_dir}"
    args=(
        --source-dir "${source_dir}"
        --data-dir "${output_dir}"
        --repo-cache "${REPO_CACHE}"
        --pyminify "${PYMINIFY}"
        --swebench-python "${SWEBENCH_PYTHON}"
        --run-id "q35gptq-python-minifier-${name}"
        --transform-workers "${TRANSFORM_WORKERS:-4}"
        --eval-workers "${EVAL_WORKERS:-4}"
        --modal-eval-timeout 1800
    )
    if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
        args+=(--prepare-only)
    fi
    if [[ "${SKIP_DETECTION:-0}" == "1" ]]; then
        args+=(--skip-detection)
    fi
    "${QWEN_PYTHON}" rebuttal_arr/swe_bench/run_python_minifier_postprocess.py \
        "${args[@]}"
done

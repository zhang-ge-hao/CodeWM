#!/usr/bin/env bash
#SBATCH --job-name=method-detect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?}"
: "${PI_WORKDIR:?PI_WORKDIR must point to the shared work directory}"
: "${METHOD_ROOT:?Set the method-scoped z-only output root}"
: "${BUNDLE_NAMES:?Set a colon-delimited experiment list}"

QWEN_PYTHON="${PI_WORKDIR}/conda/envs/qwen36/bin/python"
test -x "${QWEN_PYTHON}"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${PI_WORKDIR}/huggingface}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="${PWD}:${PWD}/rebuttal_arr/swe_bench${PYTHONPATH:+:${PYTHONPATH}}"

args=("${METHOD_ROOT}" --workers "${DETECT_WORKERS:-8}")
IFS=':' read -r -a names <<<"${BUNDLE_NAMES}"
for name in "${names[@]}"; do
    args+=(--experiment "${name}")
done
"${QWEN_PYTHON}" rebuttal_arr/swe_bench/recompute_method_detection.py "${args[@]}"

#!/bin/bash
#SBATCH --job-name=rw100-wllm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=01:00:00

set -euo pipefail

RUN_ID="${RUN_ID:-rw100-useful-v2}"
ROOT="${HOME}/project/CodeWM/rebuttal_arr/new_prototype"
PYTHON="${HOME}/conda/envs/watermarking/bin/python"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "${ROOT}"
"${PYTHON}" -u -m experiment.run --run-id "${RUN_ID}" \
    score-scheme-parallel --scheme wllm --workers 13

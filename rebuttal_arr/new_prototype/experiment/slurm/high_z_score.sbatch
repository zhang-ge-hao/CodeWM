#!/bin/bash
#SBATCH --job-name=high-z-score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00

set -euo pipefail

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
"${PYTHON}" -u -m experiment.high_z_followup \
    --followup-run-id high-z-top20-rw100-v3 \
    score \
    --overwrite

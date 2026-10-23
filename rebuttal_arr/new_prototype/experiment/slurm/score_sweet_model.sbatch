#!/bin/bash
#SBATCH --job-name=rw100-sweet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00

set -euo pipefail

MODEL_SLUG="${1:?model slug is required}"
RUN_ID="${RUN_ID:-rw100-useful-v1}"
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
echo "SWEET_START model=${MODEL_SLUG} job=${SLURM_JOB_ID} host=$(hostname) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"${PYTHON}" -u -m experiment.run --run-id "${RUN_ID}" score-sweet-model --model-slug "${MODEL_SLUG}"
echo "SWEET_END model=${MODEL_SLUG} job=${SLURM_JOB_ID} host=$(hostname) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

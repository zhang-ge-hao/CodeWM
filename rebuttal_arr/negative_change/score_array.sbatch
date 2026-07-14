#!/bin/bash
#SBATCH --job-name=codewm-neg-score
#SBATCH --array=0-284%30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00

set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${WATERMARKING_PYTHON:-${HOME}/conda/envs/watermarking/bin/python}"
RUNNER="${SCRIPT_DIR}/run.py"

echo "ARRAY_START job=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID} host=$(hostname) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"${PYTHON}" -u "${RUNNER}" score --job-index "${SLURM_ARRAY_TASK_ID}"
echo "ARRAY_END job=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID} host=$(hostname) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

#!/bin/bash
#SBATCH --job-name=codewm-neg-aggregate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${WATERMARKING_PYTHON:-${HOME}/conda/envs/watermarking/bin/python}"
RUNNER="${SCRIPT_DIR}/run.py"

echo "AGGREGATE_START job=${SLURM_JOB_ID} host=$(hostname) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"${PYTHON}" -u "${RUNNER}" aggregate --overwrite
echo "AGGREGATE_END job=${SLURM_JOB_ID} host=$(hostname) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

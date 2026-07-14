#!/bin/bash
#SBATCH --job-name=rw100-baseline
#SBATCH --array=0-123%30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00

set -euo pipefail

RUN_ID="${RUN_ID:-rw100-useful-v1}"
ROOT="${HOME}/project/CodeWM/rebuttal_arr/new_prototype"
PYTHON="/usr/bin/python"

export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${ROOT}"
echo "BASELINE_START job=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID} host=$(hostname) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"${PYTHON}" -u -m experiment.transform_cli --run-id "${RUN_ID}" --baseline-only
echo "BASELINE_END job=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID} host=$(hostname) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

#!/bin/bash
#SBATCH --job-name=dist-all1620
#SBATCH --array=0-39%40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
RUN_ID="${RUN_ID:?RUN_ID is required}"
ROOT="${HOME}/project/CodeWM/rebuttal_arr/new_prototype"
cd "${ROOT}"
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/usr/bin/python -u -m distribution_experiment.run --run-id "${RUN_ID}" transform-array \
  --array-index "${SLURM_ARRAY_TASK_ID}" --array-count 40 --workers 16

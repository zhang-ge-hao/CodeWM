#!/bin/bash
#SBATCH --job-name=dist-transform
#SBATCH --array=0-9%10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail
RUN_ID="${RUN_ID:?RUN_ID is required}"
ROOT="${HOME}/project/CodeWM/rebuttal_arr/new_prototype"
cd "${ROOT}"
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/usr/bin/python -u -m distribution_experiment.run --run-id "${RUN_ID}" transform-array \
  --array-index "${SLURM_ARRAY_TASK_ID}" --array-count 10 --workers 32

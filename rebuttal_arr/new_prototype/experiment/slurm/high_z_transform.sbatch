#!/bin/bash
#SBATCH --job-name=high-z-rw100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00

set -euo pipefail

ROOT="${HOME}/project/CodeWM/rebuttal_arr/new_prototype"
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "${ROOT}"
/usr/bin/python -u -m experiment.high_z_followup \
    --followup-run-id high-z-top20-rw100-v3 \
    transform \
    --overwrite

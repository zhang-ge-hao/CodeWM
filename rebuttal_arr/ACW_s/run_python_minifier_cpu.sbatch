#!/usr/bin/env bash
#SBATCH --job-name=acws-pymin-cpu
#SBATCH --partition=cpu
#SBATCH --array=0-3%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=rebuttal_arr/ACW_s/results/cpu-minifier-%A_%a.out
#SBATCH --error=rebuttal_arr/ACW_s/results/cpu-minifier-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

PYTHON_BIN="${ACWS_PYTHON:-${HOME}/conda/envs/watermarking/bin/python}"
SOURCE_RESULTS="${ACWS_SOURCE_RESULTS:?Set ACWS_SOURCE_RESULTS to the directory containing sample_*.json}"
OUTPUT_ROOT="${ACWS_CPU_OUTPUT_ROOT:?Set ACWS_CPU_OUTPUT_ROOT}"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"${PYTHON_BIN}" rebuttal_arr/ACW_s/python_minifier_cpu.py shard \
    --source-results "${SOURCE_RESULTS}" \
    --output-root "${OUTPUT_ROOT}" \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --shard-count 4

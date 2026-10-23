#!/usr/bin/env bash
#SBATCH --job-name=acws-pymin-gpu-agg
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=rebuttal_arr/ACW_s/results/gpu-minifier-aggregate-%j.out
#SBATCH --error=rebuttal_arr/ACW_s/results/gpu-minifier-aggregate-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

PYTHON_BIN="${ACWS_PYTHON:-${HOME}/conda/envs/watermarking/bin/python}"
SOURCE_RESULTS="${ACWS_SOURCE_RESULTS:?Set ACWS_SOURCE_RESULTS}"
CPU_ROOT="${ACWS_CPU_OUTPUT_ROOT:?Set ACWS_CPU_OUTPUT_ROOT}"
GPU_ROOT="${ACWS_GPU_OUTPUT_ROOT:?Set ACWS_GPU_OUTPUT_ROOT}"

"${PYTHON_BIN}" rebuttal_arr/ACW_s/score_python_minifier_gpu.py aggregate \
    --source-results "${SOURCE_RESULTS}" \
    --cpu-root "${CPU_ROOT}" \
    --output-root "${GPU_ROOT}"

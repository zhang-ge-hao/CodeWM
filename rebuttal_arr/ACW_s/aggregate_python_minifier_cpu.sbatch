#!/usr/bin/env bash
#SBATCH --job-name=acws-pymin-agg
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=rebuttal_arr/ACW_s/results/cpu-minifier-aggregate-%j.out
#SBATCH --error=rebuttal_arr/ACW_s/results/cpu-minifier-aggregate-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

PYTHON_BIN="${ACWS_PYTHON:-${HOME}/conda/envs/watermarking/bin/python}"
OUTPUT_ROOT="${ACWS_CPU_OUTPUT_ROOT:?Set ACWS_CPU_OUTPUT_ROOT}"

"${PYTHON_BIN}" rebuttal_arr/ACW_s/python_minifier_cpu.py aggregate \
    --output-root "${OUTPUT_ROOT}" \
    --shard-count 4

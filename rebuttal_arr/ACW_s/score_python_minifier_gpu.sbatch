#!/usr/bin/env bash
#SBATCH --job-name=acws-pymin-gpu
#SBATCH --partition=superpod-a100
#SBATCH --array=0-3%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=rebuttal_arr/ACW_s/results/gpu-minifier-%A_%a.out
#SBATCH --error=rebuttal_arr/ACW_s/results/gpu-minifier-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

PYTHON_BIN="${ACWS_PYTHON:-${HOME}/conda/envs/watermarking/bin/python}"
SOURCE_RESULTS="${ACWS_SOURCE_RESULTS:?Set ACWS_SOURCE_RESULTS}"
CPU_ROOT="${ACWS_CPU_OUTPUT_ROOT:?Set ACWS_CPU_OUTPUT_ROOT}"
GPU_ROOT="${ACWS_GPU_OUTPUT_ROOT:?Set ACWS_GPU_OUTPUT_ROOT}"
MODEL="${ACWS_MODEL:-infly/opencoder-1.5b-instruct}"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"${PYTHON_BIN}" rebuttal_arr/ACW_s/score_python_minifier_gpu.py score-shard \
    --source-results "${SOURCE_RESULTS}" \
    --cpu-root "${CPU_ROOT}" \
    --output-root "${GPU_ROOT}" \
    --model "${MODEL}" \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --shard-count 4 \
    --local-files-only

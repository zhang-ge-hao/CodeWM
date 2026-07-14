#!/usr/bin/env bash
#SBATCH --job-name=sweb-qwen36
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=data/slurm-%j.out
#SBATCH --error=data/slurm-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?}"
: "${PI_WORKDIR:?PI_WORKDIR must point to the shared work directory}"
: "${HF_HOME:?HF_HOME must point to the Hugging Face cache}"

ENV_ROOT="${CONDA_ENV_ROOT:-${PI_WORKDIR}/conda/envs}"
QWEN_PYTHON="${ENV_ROOT}/qwen36/bin/python"
SWEBENCH_PYTHON="${ENV_ROOT}/swebench/bin/python"

test -x "${QWEN_PYTHON}"
test -x "${SWEBENCH_PYTHON}"
mkdir -p data
DATA_DIR="${EXPERIMENT_DATA_DIR:-data/job-${SLURM_JOB_ID}}"
mkdir -p "${DATA_DIR}"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

RUN_ID="${EXPERIMENT_RUN_ID:-qwen36-bm25-wllm-50-${SLURM_JOB_ID}}"
exec "${QWEN_PYTHON}" run_experiment.py \
    --data-dir "${DATA_DIR}" \
    --sample-size 50 \
    --selection-seed 42 \
    --max-new-tokens 10240 \
    --modal-workers 8 \
    --run-id "${RUN_ID}" \
    --swebench-python "${SWEBENCH_PYTHON}"

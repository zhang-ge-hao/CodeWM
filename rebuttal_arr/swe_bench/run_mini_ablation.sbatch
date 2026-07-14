#!/usr/bin/env bash
#SBATCH --job-name=mini-sweb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=data/ablation-mini-swe-agent/slurm-%j.out
#SBATCH --error=data/ablation-mini-swe-agent/slurm-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?}"
: "${PI_WORKDIR:?PI_WORKDIR must point to the shared work directory}"
: "${HF_HOME:?HF_HOME must point to the Hugging Face cache}"
: "${WATERMARKING:?Set WATERMARKING to none or wllm}"
: "${EXPERIMENT_DATA_DIR:?Set the output directory}"
: "${EXPERIMENT_RUN_ID:?Set a stable run ID}"

ENV_ROOT="${CONDA_ENV_ROOT:-${PI_WORKDIR}/conda/envs}"
QWEN_PYTHON="${ENV_ROOT}/qwen36/bin/python"
SWEBENCH_PYTHON="${ENV_ROOT}/swebench/bin/python"
test -x "${QWEN_PYTHON}"
test -x "${SWEBENCH_PYTHON}"
mkdir -p "${EXPERIMENT_DATA_DIR}"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MODAL_BUILD_VALIDATION=ignore
export MSWEA_SILENT_STARTUP=1
export SWE_REX_LOG_STREAM_LEVEL=WARNING

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

ARGS=(
    --data-dir "${EXPERIMENT_DATA_DIR}"
    --sample-size "${SAMPLE_SIZE:-3}"
    --selection-seed 42
    --generation-seed "$((0x1352766))"
    --temperature 1.0
    --top-p 0.95
    --top-k 20
    --max-context-tokens "${MAX_CONTEXT_TOKENS:-200000}"
    --max-new-tokens "${MAX_NEW_TOKENS:-32768}"
    --agent-workers "${AGENT_WORKERS:-1}"
    --eval-workers "${EVAL_WORKERS:-8}"
    --modal-eval-timeout 1800
    --watermark-key 15485863
    --z-threshold 4.0
    --watermarking "${WATERMARKING}"
    --run-id "${EXPERIMENT_RUN_ID}"
    --swebench-python "${SWEBENCH_PYTHON}"
)

if [[ -n "${SELECTION_SOURCE:-}" ]]; then
    ARGS+=(--selection-source "${SELECTION_SOURCE}")
fi

if [[ "${WATERMARKING}" == "wllm" ]]; then
    : "${DELTA:?Set DELTA for a WLLM run}"
    : "${GAMMA:?Set GAMMA for a WLLM run}"
    ARGS+=(--delta "${DELTA}" --gamma "${GAMMA}")
elif [[ "${WATERMARKING}" != "none" ]]; then
    echo "Unsupported WATERMARKING=${WATERMARKING}" >&2
    exit 2
fi

exec "${QWEN_PYTHON}" run_mini_experiment.py "${ARGS[@]}"

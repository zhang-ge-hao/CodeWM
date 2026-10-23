#!/usr/bin/env bash
#SBATCH --job-name=mini-vllm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=data/ablation-mini-swe-agent-vllm/slurm-%j.out
#SBATCH --error=data/ablation-mini-swe-agent-vllm/slurm-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?}"
: "${PI_WORKDIR:?PI_WORKDIR must point to the shared work directory}"
: "${HF_HOME:?HF_HOME must point to the Hugging Face cache}"

if [[ -n "${BUNDLE_CONFIGS:-}" ]]; then
    : "${BUNDLE_OUTPUT_ROOT:?Set the root for bundled experiment outputs}"
    : "${BUNDLE_JOB_DIR:?Set the bundle job log directory}"
    job_data_dir="${BUNDLE_JOB_DIR}"
else
    : "${WATERMARKING:?Set WATERMARKING to none or wllm}"
    : "${EXPERIMENT_DATA_DIR:?Set the output directory}"
    : "${EXPERIMENT_RUN_ID:?Set the run ID}"
    job_data_dir="${EXPERIMENT_DATA_DIR}"
fi

ENV_ROOT="${CONDA_ENV_ROOT:-${PI_WORKDIR}/conda/envs}"
VLLM_BIN="${ENV_ROOT}/vllm019/bin/vllm"
QWEN_PYTHON="${ENV_ROOT}/qwen36/bin/python"
SWEBENCH_PYTHON="${ENV_ROOT}/swebench/bin/python"
if [[ -n "${VLLM_PORT:-}" ]]; then
    PORT="${VLLM_PORT}"
else
    PORT="$(${QWEN_PYTHON} -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
fi
MODEL_ID="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
SERVED_MODEL="Qwen3.5-35B-A3B-GPTQ-Int4"
mkdir -p "${job_data_dir}"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export MODAL_BUILD_VALIDATION=ignore
export MSWEA_SILENT_STARTUP=1
export SWE_REX_LOG_STREAM_LEVEL=WARNING
export PYTHONPATH="${PWD}:${PWD}/rebuttal_arr/swe_bench${PYTHONPATH:+:${PYTHONPATH}}"

server_log="${job_data_dir}/vllm-server.log"
MODEL_PATH="$(${QWEN_PYTHON} -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Qwen/Qwen3.5-35B-A3B-GPTQ-Int4", local_files_only=True))')"
"${VLLM_BIN}" serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --quantization moe_wna16 \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.94 \
    --max-model-len 200000 \
    --max-num-seqs 25 \
    --enable-prefix-caching \
    --language-model-only \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --logits-processors vllm_wllm:VLLMWatermarkLogitsProcessor \
    >"${server_log}" 2>&1 &
server_pid=$!
cleanup() {
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

while ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null; do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        tail -200 "${server_log}" >&2
        exit 1
    fi
    sleep 2
done

run_one() {
    local watermarking="$1"
    local output="$2"
    local run_id="$3"
    local delta="${4:-}"
    local gamma="${5:-}"
    mkdir -p "${output}"
    local args=(
        --data-dir "${output}"
        --sample-size "${SAMPLE_SIZE:-10}"
        --selection-seed 42
        --generation-seed "$((0x1352766))"
        --model-id "${MODEL_ID}"
        --served-model-name "${SERVED_MODEL}"
        --selection-source "${SELECTION_SOURCE:-data/selection.json}"
        --temperature 1.0
        --top-p 0.95
        --top-k 20
        --max-new-tokens "${MAX_NEW_TOKENS:-10000}"
        --agent-workers "${AGENT_WORKERS:-25}"
        --environment-start-attempts "${ENVIRONMENT_START_ATTEMPTS:-5}"
        --agent-wall-time-limit-seconds "${AGENT_WALL_TIME_LIMIT_SECONDS:-1800}"
        --eval-workers "${EVAL_WORKERS:-8}"
        --modal-eval-timeout 1800
        --watermark-key 15485863
        --z-threshold 4.0
        --watermarking "${watermarking}"
        --vllm-base-url "http://127.0.0.1:${PORT}"
        --run-id "${run_id}"
        --swebench-python "${SWEBENCH_PYTHON}"
    )
    if [[ "${watermarking}" == "wllm" ]]; then
        args+=(--delta "${delta}" --gamma "${gamma}")
    fi
    "${QWEN_PYTHON}" run_mini_vllm_experiment.py "${args[@]}"
}

if [[ -n "${BUNDLE_CONFIGS:-}" ]]; then
    IFS=';' read -r -a bundle_items <<<"${BUNDLE_CONFIGS}"
    for item in "${bundle_items[@]}"; do
        IFS=':' read -r delta gamma <<<"${item}"
        name="wllm-delta-${delta}-gamma-${gamma}-ngram-5"
        run_one \
            wllm \
            "${BUNDLE_OUTPUT_ROOT}/${name}" \
            "q35gptq-${name}" \
            "${delta}" \
            "${gamma}"
    done
else
    run_one \
        "${WATERMARKING}" \
        "${EXPERIMENT_DATA_DIR}" \
        "${EXPERIMENT_RUN_ID}" \
        "${DELTA:-}" \
        "${GAMMA:-}"
fi

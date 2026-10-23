#!/usr/bin/env bash
# Run the non-watermarked and strongest-WLLM gates concurrently on four GPUs.
# This script is intended for an existing four-GPU Slurm allocation.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

: "${PI_WORKDIR:?PI_WORKDIR must point to the shared work directory}"
: "${HF_HOME:?HF_HOME must point to the Hugging Face cache}"
ENV_ROOT="${CONDA_ENV_ROOT:-${PI_WORKDIR}/conda/envs}"
VLLM_BIN="${ENV_ROOT}/vllm019/bin/vllm"
QWEN_PYTHON="${ENV_ROOT}/qwen36/bin/python"
SWEBENCH_PYTHON="${ENV_ROOT}/swebench/bin/python"
DATA_ROOT="${DATA_ROOT:-data/ablation-mini-swe-agent-vllm/gate-10}"
SELECTION_SOURCE="${SELECTION_SOURCE:-data/selection.json}"
SAMPLE_SIZE="${SAMPLE_SIZE:-10}"
AGENT_WORKERS="${AGENT_WORKERS:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10000}"
PORT_BASE="${VLLM_PORT_BASE:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}"
NON_WM_PORT="${NON_WM_PORT:-${PORT_BASE}}"
WLLM_PORT="${WLLM_PORT:-$((PORT_BASE + 1))}"
mkdir -p "${DATA_ROOT}/non-wm" "${DATA_ROOT}/wllm-delta-4-gamma-0.5-ngram-5"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MODAL_BUILD_VALIDATION=ignore
export MSWEA_SILENT_STARTUP=1
export SWE_REX_LOG_STREAM_LEVEL=WARNING
export PYTHONPATH="${PWD}:$(cd ../.. && pwd)${PYTHONPATH:+:${PYTHONPATH}}"

server_pids=()
controller_pids=()
heartbeat_pid=""
cleanup() {
    for pid in "${controller_pids[@]}" "${server_pids[@]}" "${heartbeat_pid}"; do
        [[ -n "${pid:-}" ]] && kill "${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Preserve a small regular compute load during imports and Modal-only tails,
# when both vLLM services are resident but may have no inference requests.
CUDA_VISIBLE_DEVICES="0,1,2,3" "${ENV_ROOT}/vllm019/bin/python" -c '
import time
import torch
devices = [torch.device(f"cuda:{index}") for index in range(4)]
matrices = [torch.randn((8192, 8192), device=device) for device in devices]
while True:
    products = [torch.mm(matrix, matrix) for matrix in matrices]
    for device in devices:
        with torch.cuda.device(device):
            torch.cuda.synchronize()
    del products
    time.sleep(1)
' >"${DATA_ROOT}/gpu-heartbeat.log" 2>&1 &
heartbeat_pid=$!

start_server() {
    local devices="$1"
    local port="$2"
    local log="$3"
    CUDA_VISIBLE_DEVICES="${devices}" "${VLLM_BIN}" serve Qwen/Qwen3.6-35B-A3B \
        --served-model-name Qwen3.6-35B-A3B \
        --host 127.0.0.1 \
        --port "${port}" \
        --tensor-parallel-size 2 \
        --max-model-len 200000 \
        --enforce-eager \
        --language-model-only \
        --reasoning-parser qwen3 \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --logits-processors vllm_wllm:VLLMWatermarkLogitsProcessor \
        >"${log}" 2>&1 &
    server_pids+=("$!")
}

start_server "0,1" "${NON_WM_PORT}" "${DATA_ROOT}/non-wm/vllm-server.log"
start_server "2,3" "${WLLM_PORT}" "${DATA_ROOT}/wllm-delta-4-gamma-0.5-ngram-5/vllm-server.log"

ports=("${NON_WM_PORT}" "${WLLM_PORT}")
logs=(
    "${DATA_ROOT}/non-wm/vllm-server.log"
    "${DATA_ROOT}/wllm-delta-4-gamma-0.5-ngram-5/vllm-server.log"
)
for index in 0 1; do
    port=${ports[$index]}
    log=${logs[$index]}
    pid=${server_pids[$index]}
    while ! curl -fsS "http://127.0.0.1:${port}/health" >/dev/null; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            tail -200 "${log}" >&2
            exit 1
        fi
        sleep 2
    done
done
echo "Both vLLM services are healthy on ports ${NON_WM_PORT} and ${WLLM_PORT}" >&2

common=(
    --sample-size "${SAMPLE_SIZE}"
    --selection-seed 42
    --generation-seed "$((0x1352766))"
    --selection-source "${SELECTION_SOURCE}"
    --temperature 1.0
    --top-p 0.95
    --top-k 20
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --agent-workers "${AGENT_WORKERS}"
    --eval-workers 8
    --modal-eval-timeout 1800
    --watermark-key 15485863
    --z-threshold 4.0
    --swebench-python "${SWEBENCH_PYTHON}"
)

"${QWEN_PYTHON}" run_mini_vllm_experiment.py \
    "${common[@]}" \
    --data-dir "${DATA_ROOT}/non-wm" \
    --watermarking none \
    --vllm-base-url "http://127.0.0.1:${NON_WM_PORT}" \
    --run-id mini-vllm-gate-non-wm \
    >"${DATA_ROOT}/non-wm/controller.out" 2>"${DATA_ROOT}/non-wm/controller.err" &
controller_pids+=("$!")

"${QWEN_PYTHON}" run_mini_vllm_experiment.py \
    "${common[@]}" \
    --data-dir "${DATA_ROOT}/wllm-delta-4-gamma-0.5-ngram-5" \
    --watermarking wllm \
    --delta 4 \
    --gamma 0.5 \
    --vllm-base-url "http://127.0.0.1:${WLLM_PORT}" \
    --run-id mini-vllm-gate-wllm-d4-g0.5 \
    >"${DATA_ROOT}/wllm-delta-4-gamma-0.5-ngram-5/controller.out" \
    2>"${DATA_ROOT}/wllm-delta-4-gamma-0.5-ngram-5/controller.err" &
controller_pids+=("$!")

status=0
for pid in "${controller_pids[@]}"; do
    wait "${pid}" || status=$?
done
exit "${status}"

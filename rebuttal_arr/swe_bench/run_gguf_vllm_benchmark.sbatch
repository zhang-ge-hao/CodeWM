#!/usr/bin/env bash
#SBATCH --job-name=qwen36-gguf-bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=rebuttal_arr/swe_bench/data/ablation-mini-swe-agent-vllm/gguf-benchmark/slurm-%j.out
#SBATCH --error=rebuttal_arr/swe_bench/data/ablation-mini-swe-agent-vllm/gguf-benchmark/slurm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

PI_WORKDIR="${PI_WORKDIR:-${HOME}/.cache/codewm}"
HF_HOME="${HF_HOME:-${PI_WORKDIR}/huggingface}"
ENV_ROOT="${CONDA_ENV_ROOT:-${PI_WORKDIR}/conda/envs}"
VLLM_BIN="${ENV_ROOT}/vllm019/bin/vllm"
VLLM_PYTHON="${ENV_ROOT}/vllm019/bin/python"
QWEN_PYTHON="${ENV_ROOT}/qwen36/bin/python"
OUTPUT_ROOT="${OUTPUT_ROOT:-rebuttal_arr/swe_bench/data/ablation-mini-swe-agent-vllm/gguf-benchmark/job-${SLURM_JOB_ID}}"
PORT="${VLLM_PORT:-$((23000 + SLURM_JOB_ID % 10000))}"
SERVED_MODEL="Qwen3.6-35B-A3B-GGUF-Q4KM"
TRAJECTORY_ROOT="${TRAJECTORY_ROOT:-rebuttal_arr/swe_bench/data/ablation-mini-swe-agent}"
mkdir -p "${OUTPUT_ROOT}"

export HF_HOME PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PWD}/rebuttal_arr/swe_bench:${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

GGUF_PATH="$(${QWEN_PYTHON} -c 'from huggingface_hub import hf_hub_download; print(hf_hub_download("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", local_files_only=True))')"
MODEL_TAG="${MODEL_TAG:-${GGUF_PATH}}"
printf 'GGUF_PATH=%s\n' "${GGUF_PATH}" | tee "${OUTPUT_ROOT}/settings.txt"
printf 'MODEL_TAG=%s\n' "${MODEL_TAG}" | tee -a "${OUTPUT_ROOT}/settings.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "${OUTPUT_ROOT}/settings.txt"
"${VLLM_PYTHON}" -c 'import importlib.metadata as m; print("vllm=" + m.version("vllm"))' | tee -a "${OUTPUT_ROOT}/settings.txt"

"${VLLM_BIN}" serve "${MODEL_TAG}" \
    --served-model-name "${SERVED_MODEL}" \
    --tokenizer Qwen/Qwen3.6-35B-A3B \
    --hf-config-path Qwen/Qwen3.6-35B-A3B \
    --quantization gguf \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.94 \
    --max-model-len 200000 \
    --max-num-seqs 32 \
    --language-model-only \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --logits-processors vllm_wllm:VLLMWatermarkLogitsProcessor \
    >"${OUTPUT_ROOT}/vllm-server.log" 2>&1 &
server_pid=$!
cleanup() {
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

while ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null; do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        tail -300 "${OUTPUT_ROOT}/vllm-server.log" >&2
        exit 1
    fi
    sleep 2
done
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tee "${OUTPUT_ROOT}/server-memory.csv"

common=(
    --base-url "http://127.0.0.1:${PORT}"
    --model "${SERVED_MODEL}"
    --trajectory-root "${TRAJECTORY_ROOT}"
    --temperature 1.0
    --top-p 0.95
    --top-k 20
)

"${VLLM_PYTHON}" rebuttal_arr/swe_bench/benchmark_vllm_concurrency.py \
    "${common[@]}" --profile initial --max-tokens 512 \
    --concurrency 1 2 4 8 12 16 24 32 \
    --output "${OUTPUT_ROOT}/initial-context.json" | tee "${OUTPUT_ROOT}/initial-context.log"

"${VLLM_PYTHON}" rebuttal_arr/swe_bench/benchmark_vllm_concurrency.py \
    "${common[@]}" --profile long --max-tokens 512 \
    --concurrency 1 2 4 8 12 16 24 32 \
    --output "${OUTPUT_ROOT}/long-context.json" | tee "${OUTPUT_ROOT}/long-context.log"

# Confirm the request-level WLLM processor remains compatible with GGUF weights.
"${VLLM_PYTHON}" rebuttal_arr/swe_bench/benchmark_vllm_concurrency.py \
    "${common[@]}" --profile initial --max-tokens 512 --concurrency 4 --wllm \
    --output "${OUTPUT_ROOT}/wllm-smoke.json" | tee "${OUTPUT_ROOT}/wllm-smoke.log"

nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader | tee "${OUTPUT_ROOT}/final-gpu.csv"

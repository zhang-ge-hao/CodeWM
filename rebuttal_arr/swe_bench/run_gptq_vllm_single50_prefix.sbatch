#!/usr/bin/env bash
#SBATCH --job-name=qwen35-gptq-1gpu-50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=rebuttal_arr/swe_bench/data/ablation-mini-swe-agent-vllm/gptq-single50-prefix/slurm-%j.out
#SBATCH --error=rebuttal_arr/swe_bench/data/ablation-mini-swe-agent-vllm/gptq-single50-prefix/slurm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

PI_WORKDIR="${PI_WORKDIR:-${HOME}/.cache/codewm}"
HF_HOME="${HF_HOME:-${PI_WORKDIR}/huggingface}"
ENV_ROOT="${CONDA_ENV_ROOT:-${PI_WORKDIR}/conda/envs}"
VLLM_BIN="${ENV_ROOT}/vllm019/bin/vllm"
VLLM_PYTHON="${ENV_ROOT}/vllm019/bin/python"
CLIENT_PYTHON="${ENV_ROOT}/qwen36/bin/python"
SERVED_MODEL="Qwen3.5-35B-A3B-GPTQ-Int4"
OUTPUT_ROOT="${OUTPUT_ROOT:-rebuttal_arr/swe_bench/data/ablation-mini-swe-agent-vllm/gptq-single50-prefix/job-${SLURM_JOB_ID}}"
PORT="${VLLM_PORT:-$((24000 + SLURM_JOB_ID % 10000))}"
TRAJECTORY_ROOT="${TRAJECTORY_ROOT:-rebuttal_arr/swe_bench/data/ablation-mini-swe-agent}"
SELECTION="${SELECTION:-rebuttal_arr/swe_bench/data/ablation-mini-swe-agent/full-50/non-wm/selection.json}"
mkdir -p "${OUTPUT_ROOT}"

export HF_HOME PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PWD}/rebuttal_arr/swe_bench:${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_PATH="$(${CLIENT_PYTHON} -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Qwen/Qwen3.5-35B-A3B-GPTQ-Int4", local_files_only=True))')"
VOCAB_SIZE="$(${CLIENT_PYTHON} -c 'from transformers import AutoTokenizer; print(len(AutoTokenizer.from_pretrained("Qwen/Qwen3.5-35B-A3B-GPTQ-Int4", local_files_only=True).get_vocab()))')"
printf 'MODEL_PATH=%s\nVOCAB_SIZE=%s\nGPUS=1\nMAX_NUM_SEQS=50\nPREFIX_CACHING=true\nSELECTION=%s\n' \
    "${MODEL_PATH}" "${VOCAB_SIZE}" "${SELECTION}" | tee "${OUTPUT_ROOT}/settings.txt"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | tee -a "${OUTPUT_ROOT}/settings.txt"
"${VLLM_PYTHON}" -c 'import importlib.metadata as m; print("vllm=" + m.version("vllm"))' | tee -a "${OUTPUT_ROOT}/settings.txt"

"${VLLM_BIN}" serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --quantization moe_wna16 \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.94 \
    --max-model-len 200000 \
    --max-num-seqs 50 \
    --enable-prefix-caching \
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
    --selection "${SELECTION}"
    --temperature 1.0
    --top-p 0.95
    --top-k 20
    --vocab-size "${VOCAB_SIZE}"
)

run_profile() {
    local profile="$1"
    local label="$2"
    shift 2
    "${CLIENT_PYTHON}" rebuttal_arr/swe_bench/benchmark_vllm_concurrency.py \
        "${common[@]}" --profile "${profile}" --max-tokens 512 \
        --concurrency 50 25 10 "$@" \
        --output "${OUTPUT_ROOT}/${label}.json" | tee "${OUTPUT_ROOT}/${label}.log"
}

run_profile initial non-wm-initial
run_profile long non-wm-long
run_profile initial wllm-initial --wllm
run_profile long wllm-long --wllm

nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader | tee "${OUTPUT_ROOT}/final-gpu.csv"

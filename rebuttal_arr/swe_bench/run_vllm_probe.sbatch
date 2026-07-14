#!/usr/bin/env bash
#SBATCH --job-name=wllm-vllm-probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=rebuttal_arr/swe_bench/data/ablation-mini-swe-agent-vllm/probe/slurm-%j.out
#SBATCH --error=rebuttal_arr/swe_bench/data/ablation-mini-swe-agent-vllm/probe/slurm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

PI_WORKDIR="${PI_WORKDIR:-${HOME}/.cache/codewm}"
HF_HOME="${HF_HOME:-${PI_WORKDIR}/huggingface}"
ENV_ROOT="${PI_WORKDIR}/conda/envs"
PORT="${VLLM_PORT:-$((18000 + SLURM_JOB_ID % 1000))}"
OUT_DIR="rebuttal_arr/swe_bench/data/ablation-mini-swe-agent-vllm/probe/job-${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"

export HF_HOME PI_WORKDIR
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}:${PWD}/rebuttal_arr/swe_bench"

"${ENV_ROOT}/vllm019/bin/vllm" serve Qwen/Qwen3.6-35B-A3B \
    --served-model-name Qwen3.6-35B-A3B \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --enforce-eager \
    --language-model-only \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --logits-processors vllm_wllm:VLLMWatermarkLogitsProcessor \
    >"${OUT_DIR}/vllm-server.log" 2>&1 &
server_pid=$!
cleanup() {
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

while ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null; do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        tail -200 "${OUT_DIR}/vllm-server.log" >&2
        exit 1
    fi
    sleep 2
done

"${ENV_ROOT}/qwen36/bin/python" rebuttal_arr/swe_bench/probe_vllm_wllm.py \
    --base-url "http://127.0.0.1:${PORT}" \
    --output "${OUT_DIR}/result.json"

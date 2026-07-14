#!/usr/bin/env bash
#SBATCH --job-name=mini-vllm-dual
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=rebuttal_arr/swe_bench/data/ablation-mini-swe-agent-vllm/slurm-dual-%j.out
#SBATCH --error=rebuttal_arr/swe_bench/data/ablation-mini-swe-agent-vllm/slurm-dual-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

export PI_WORKDIR="${PI_WORKDIR:-${HOME}/.cache/codewm}"
export HF_HOME="${HF_HOME:-${PI_WORKDIR}/huggingface}"
export SAMPLE_SIZE="${SAMPLE_SIZE:-10}"
export AGENT_WORKERS="${AGENT_WORKERS:-2}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10000}"
export DATA_ROOT="${DATA_ROOT:-data/ablation-mini-swe-agent-vllm/gate-${SAMPLE_SIZE}-job-${SLURM_JOB_ID}}"
export SELECTION_SOURCE="${SELECTION_SOURCE:-data/selection.json}"

cd rebuttal_arr/swe_bench
exec ./run_vllm_dual_gate.sh

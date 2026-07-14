#!/bin/bash
set -euo pipefail

RUN_ID="${RUN_ID:-rw100-useful-v2}"
LLAMA_JOB="${LLAMA_JOB:?completed or running SWEET Llama array job id is required}"
ROOT="${HOME}/project/CodeWM/rebuttal_arr/new_prototype"
SLURM_DIR="${ROOT}/experiment/slurm"
LOG_DIR="${ROOT}/data/watermark_attack/${RUN_ID}/slurm"

download_job=$(sbatch --parsable --output="${LOG_DIR}/deepseek-download-%j.out" --error="${LOG_DIR}/deepseek-download-%j.err" "${SLURM_DIR}/download_deepseek.sbatch")
deepseek_job=$(sbatch --parsable --dependency=afterok:"${download_job}" --cpus-per-task=4 --mem=96G --time=01:00:00 --output="${LOG_DIR}/sweet-deepseek-%j.out" --error="${LOG_DIR}/sweet-deepseek-%j.err" --export=ALL,RUN_ID="${RUN_ID}" "${SLURM_DIR}/score_sweet_model.sbatch" DSCoderBase33B)
aggregate_job=$(sbatch --parsable --dependency=afterok:"${LLAMA_JOB}:${deepseek_job}" --output="${LOG_DIR}/aggregate-%j.out" --error="${LOG_DIR}/aggregate-%j.err" --export=ALL,RUN_ID="${RUN_ID}" "${SLURM_DIR}/aggregate.sbatch")

printf 'deepseek_download=%s\ndeepseek_score=%s\naggregate=%s\n' \
  "${download_job}" "${deepseek_job}" "${aggregate_job}"

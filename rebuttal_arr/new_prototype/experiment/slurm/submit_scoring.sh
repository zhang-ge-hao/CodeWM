#!/bin/bash
set -euo pipefail

RUN_ID="${RUN_ID:-rw100-useful-v2}"
ROOT="${HOME}/project/CodeWM/rebuttal_arr/new_prototype"
SLURM_DIR="${ROOT}/experiment/slurm"
LOG_DIR="${ROOT}/data/watermark_attack/${RUN_ID}/slurm"

mkdir -p "${LOG_DIR}"

validate_job=$(sbatch --parsable --output="${LOG_DIR}/validate-%j.out" --error="${LOG_DIR}/validate-%j.err" --export=ALL,RUN_ID="${RUN_ID}" "${SLURM_DIR}/validate_transforms.sbatch")
wllm_job=$(sbatch --parsable --dependency=afterok:"${validate_job}" --output="${LOG_DIR}/wllm-%j.out" --error="${LOG_DIR}/wllm-%j.err" --export=ALL,RUN_ID="${RUN_ID}" "${SLURM_DIR}/score_wllm_parallel.sbatch")
synthid_job=$(sbatch --parsable --dependency=afterok:"${validate_job}" --output="${LOG_DIR}/synthid-%A_%a.out" --error="${LOG_DIR}/synthid-%A_%a.err" --export=ALL,RUN_ID="${RUN_ID}" "${SLURM_DIR}/score_synthid_array.sbatch")
sweet_llama_job=$(sbatch --parsable --array=0-11%12 --dependency=afterok:"${validate_job}" --output="${LOG_DIR}/sweet-llama-%A_%a.out" --error="${LOG_DIR}/sweet-llama-%A_%a.err" --export=ALL,RUN_ID="${RUN_ID}" "${SLURM_DIR}/score_sweet_shard.sbatch")
sweet_deepseek_job=$(sbatch --parsable --dependency=afterok:"${validate_job}" --cpus-per-task=4 --mem=96G --time=01:00:00 --output="${LOG_DIR}/sweet-deepseek-%j.out" --error="${LOG_DIR}/sweet-deepseek-%j.err" --export=ALL,RUN_ID="${RUN_ID}" "${SLURM_DIR}/score_sweet_model.sbatch" DSCoderBase33B)
aggregate_job=$(sbatch --parsable --dependency=afterok:"${wllm_job}:${synthid_job}:${sweet_llama_job}:${sweet_deepseek_job}" --output="${LOG_DIR}/aggregate-%j.out" --error="${LOG_DIR}/aggregate-%j.err" --export=ALL,RUN_ID="${RUN_ID}" "${SLURM_DIR}/aggregate.sbatch")

printf 'validate=%s\nwllm=%s\nsynthid=%s\nsweet_llama=%s\nsweet_deepseek=%s\naggregate=%s\n' \
  "${validate_job}" "${wllm_job}" "${synthid_job}" "${sweet_llama_job}" "${sweet_deepseek_job}" "${aggregate_job}"

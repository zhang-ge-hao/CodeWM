#!/bin/bash
#SBATCH --job-name=dist-all-wllm
#SBATCH --array=0-7%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
RUN_ID="${RUN_ID:?RUN_ID is required}"
ROOT="${HOME}/project/CodeWM/rebuttal_arr/new_prototype"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
cd "${ROOT}"
"${HOME}/conda/envs/watermarking/bin/python" -u -m distribution_experiment.run --run-id "${RUN_ID}" score-group \
  --scheme wllm --shard-index "${SLURM_ARRAY_TASK_ID}" --shard-count 8

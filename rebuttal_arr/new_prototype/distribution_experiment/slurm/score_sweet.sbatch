#!/bin/bash
#SBATCH --job-name=dist-sweet
#SBATCH --array=0-7%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00

set -euo pipefail
RUN_ID="${RUN_ID:?RUN_ID is required}"
ROOT="${HOME}/project/CodeWM/rebuttal_arr/new_prototype"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
cd "${ROOT}"
if (( SLURM_ARRAY_TASK_ID < 4 )); then
  MODEL=Llama31Instruct8B
  INDEX="${SLURM_ARRAY_TASK_ID}"
else
  MODEL=DSCoderBase33B
  INDEX="$((SLURM_ARRAY_TASK_ID - 4))"
fi
"${HOME}/conda/envs/watermarking/bin/python" -u -m distribution_experiment.run --run-id "${RUN_ID}" score-group \
  --scheme sweet --model-slug "${MODEL}" --shard-index "${INDEX}" --shard-count 4

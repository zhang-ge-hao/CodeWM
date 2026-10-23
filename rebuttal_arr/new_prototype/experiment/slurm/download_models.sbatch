#!/bin/bash
#SBATCH --job-name=rw100-model-cache
#SBATCH --array=0-1%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00

set -euo pipefail

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

CLI="${HOME}/conda/envs/watermarking/bin/huggingface-cli"
MODELS=(
  "meta-llama/Meta-Llama-3.1-8B-Instruct"
  "deepseek-ai/deepseek-coder-33b-base"
)
MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"

echo "DOWNLOAD_START model=${MODEL} job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} host=$(hostname) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"${CLI}" download "${MODEL}"
echo "DOWNLOAD_END model=${MODEL} job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} host=$(hostname) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

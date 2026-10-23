#!/bin/bash
#SBATCH --job-name=deepseek-cache
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00

set -euo pipefail

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_DISABLE_TELEMETRY=1

CLI="${HOME}/conda/envs/watermarking/bin/huggingface-cli"
"${CLI}" download deepseek-ai/deepseek-coder-33b-base \
    --include "*.safetensors" "*.json" "tokenizer*" \
    --max-workers 8

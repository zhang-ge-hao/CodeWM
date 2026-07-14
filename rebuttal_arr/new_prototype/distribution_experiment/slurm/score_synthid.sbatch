#!/bin/bash
#SBATCH --job-name=dist-synthid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
RUN_ID="${RUN_ID:?RUN_ID is required}"
ROOT="${HOME}/project/CodeWM/rebuttal_arr/new_prototype"
PYTHON="${SYNTHID_PYTHON:-${HOME}/conda/envs/codewm-synth250/bin/python}"
BASE_SITE="${HOME}/conda/envs/watermarking/lib/python3.10/site-packages"
NVIDIA_LIBS=$(find "${BASE_SITE}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -print | paste -sd: -)
export LD_LIBRARY_PATH="${BASE_SITE}/torch/lib:${NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
cd "${ROOT}"
"${PYTHON}" -u -m distribution_experiment.run --run-id "${RUN_ID}" score-group \
  --scheme synthid --shard-index 0 --shard-count 1 --synthid-device cuda

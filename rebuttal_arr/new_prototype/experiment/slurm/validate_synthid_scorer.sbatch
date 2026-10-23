#!/bin/bash
#SBATCH --job-name=validate-synthid-scorer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00

set -euo pipefail

RUN_ID="${RUN_ID:-rw100-useful-v1}"
CONFIG_INDEX="${CONFIG_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
ROOT="${HOME}/project/CodeWM/rebuttal_arr/new_prototype"
PYTHON="${SYNTHID_PYTHON:-${HOME}/conda/envs/codewm-synth250/bin/python}"
BASE_SITE="${HOME}/conda/envs/watermarking/lib/python3.10/site-packages"
NVIDIA_LIBS=$(find "${BASE_SITE}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -print | paste -sd: -)
export LD_LIBRARY_PATH="${BASE_SITE}/torch/lib:${NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0

cd "${ROOT}"
"${PYTHON}" -c 'import torch, transformers; assert torch.__version__.startswith("2.5.0"), torch.__version__; assert transformers.__version__ == "4.47.0", transformers.__version__'
"${PYTHON}" -u -m experiment.run \
  --run-id "${RUN_ID}" \
  validate-scorer \
  --scheme synthid \
  --index "${CONFIG_INDEX}" \
  --synthid-device cuda

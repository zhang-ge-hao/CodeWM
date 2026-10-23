#!/bin/bash
#SBATCH --job-name=rw100-synth-env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00

set -euo pipefail

BASE="${HOME}/conda/envs/watermarking/bin/python"
ENV_ROOT="${SYNTHID_ENV_ROOT:-${HOME}/conda/envs/codewm-synth250}"

if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
  "${BASE}" -m venv --system-site-packages "${ENV_ROOT}"
fi
"${ENV_ROOT}/bin/python" -m pip install --no-deps 'torch==2.5.0'
"${ENV_ROOT}/bin/python" -c 'import torch, transformers; print("torch", torch.__version__, "transformers", transformers.__version__); assert torch.__version__.startswith("2.5.0"); assert transformers.__version__ == "4.47.0"'

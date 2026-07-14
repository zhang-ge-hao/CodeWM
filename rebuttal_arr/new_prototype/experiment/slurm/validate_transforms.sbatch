#!/bin/bash
#SBATCH --job-name=rw100-validate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00

set -euo pipefail

RUN_ID="${RUN_ID:-rw100-useful-v2}"
ROOT="${HOME}/project/CodeWM/rebuttal_arr/new_prototype"
PYTHON="${HOME}/conda/envs/watermarking/bin/python"

cd "${ROOT}"
"${PYTHON}" -u -m experiment.run --run-id "${RUN_ID}" validate-transforms

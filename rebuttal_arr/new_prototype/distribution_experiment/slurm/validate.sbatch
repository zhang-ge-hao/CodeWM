#!/bin/bash
#SBATCH --job-name=dist-validate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -euo pipefail
RUN_ID="${RUN_ID:?RUN_ID is required}"
ROOT="${HOME}/project/CodeWM/rebuttal_arr/new_prototype"
cd "${ROOT}"
"${HOME}/conda/envs/watermarking/bin/python" -u -m distribution_experiment.run --run-id "${RUN_ID}" validate

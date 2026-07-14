#!/bin/bash
set -euo pipefail
RUN_ID="${RUN_ID:-rw100-z4-sample100-v2}"
ROOT="${HOME}/project/CodeWM/rebuttal_arr/new_prototype"
SLURM="${ROOT}/distribution_experiment/slurm"
RUN_ROOT="${ROOT}/data/distribution_consistency/${RUN_ID}"
LOG="${RUN_ROOT}/slurm"
[[ "${RUN_ID}" == "rw100-z4-sample100-v2" ]] || { echo "unapproved run id" >&2; exit 1; }
[[ -f "${RUN_ROOT}/manifest.json" ]] || { echo "missing manifest" >&2; exit 1; }
[[ ! -e "${RUN_ROOT}/transforms" && ! -e "${RUN_ROOT}/scores" && ! -e "${RUN_ROOT}/summary.json" ]] || { echo "refusing overwrite" >&2; exit 1; }
mkdir -p "${LOG}"
transform=$(sbatch --parsable --output="${LOG}/transform-%A_%a.out" --error="${LOG}/transform-%A_%a.err" --export=ALL,RUN_ID="${RUN_ID}" "${SLURM}/transform.sbatch")
validate=$(sbatch --parsable --dependency=afterok:"${transform}" --output="${LOG}/validate-%j.out" --error="${LOG}/validate-%j.err" --export=ALL,RUN_ID="${RUN_ID}" "${SLURM}/validate.sbatch")
wllm=$(sbatch --parsable --dependency=afterok:"${validate}" --output="${LOG}/wllm-%A_%a.out" --error="${LOG}/wllm-%A_%a.err" --export=ALL,RUN_ID="${RUN_ID}" "${SLURM}/score_wllm.sbatch")
sweet=$(sbatch --parsable --dependency=afterok:"${validate}" --output="${LOG}/sweet-%A_%a.out" --error="${LOG}/sweet-%A_%a.err" --export=ALL,RUN_ID="${RUN_ID}" "${SLURM}/score_sweet.sbatch")
synthid=$(sbatch --parsable --dependency=afterok:"${validate}" --output="${LOG}/synthid-%j.out" --error="${LOG}/synthid-%j.err" --export=ALL,RUN_ID="${RUN_ID}" "${SLURM}/score_synthid.sbatch")
aggregate=$(sbatch --parsable --dependency=afterok:"${wllm}:${sweet}:${synthid}" --output="${LOG}/aggregate-%j.out" --error="${LOG}/aggregate-%j.err" --export=ALL,RUN_ID="${RUN_ID}" "${SLURM}/aggregate.sbatch")
printf 'transform=%s\nvalidate=%s\nwllm=%s\nsweet=%s\nsynthid=%s\naggregate=%s\n' "${transform}" "${validate}" "${wllm}" "${sweet}" "${synthid}" "${aggregate}"

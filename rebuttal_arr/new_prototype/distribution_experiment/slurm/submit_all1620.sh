#!/bin/bash
set -euo pipefail
RUN_ID="${RUN_ID:-rw100-z4-all1620-random-seeds-v1}"
ROOT="${HOME}/project/CodeWM/rebuttal_arr/new_prototype"
SLURM="${ROOT}/distribution_experiment/slurm"
RUN_ROOT="${ROOT}/data/distribution_consistency/${RUN_ID}"
LOG="${RUN_ROOT}/slurm"
[[ "${RUN_ID}" == "rw100-z4-all1620-random-seeds-v1" ]] || { echo "unapproved run id" >&2; exit 1; }
cd "${ROOT}"
"${HOME}/conda/envs/watermarking/bin/python" -u -m distribution_experiment.all_cases --run-id "${RUN_ID}" manifest
mkdir -p "${LOG}"
transform=$(sbatch --parsable --output="${LOG}/transform-%A_%a.out" --error="${LOG}/transform-%A_%a.err" --export=ALL,RUN_ID="${RUN_ID}" "${SLURM}/transform_all1620.sbatch")
validate=$(sbatch --parsable --dependency=afterok:"${transform}" --output="${LOG}/validate-%j.out" --error="${LOG}/validate-%j.err" --export=ALL,RUN_ID="${RUN_ID}" "${SLURM}/validate_all1620.sbatch")
wllm=$(sbatch --parsable --dependency=afterok:"${validate}" --output="${LOG}/wllm-%A_%a.out" --error="${LOG}/wllm-%A_%a.err" --export=ALL,RUN_ID="${RUN_ID}" "${SLURM}/score_wllm_all1620.sbatch")
sweet=$(sbatch --parsable --dependency=afterok:"${validate}" --output="${LOG}/sweet-%A_%a.out" --error="${LOG}/sweet-%A_%a.err" --export=ALL,RUN_ID="${RUN_ID}" "${SLURM}/score_sweet.sbatch")
synthid=$(sbatch --parsable --dependency=afterok:"${validate}" --output="${LOG}/synthid-%j.out" --error="${LOG}/synthid-%j.err" --export=ALL,RUN_ID="${RUN_ID}" "${SLURM}/score_synthid.sbatch")
aggregate=$(sbatch --parsable --dependency=afterok:"${wllm}:${sweet}:${synthid}" --output="${LOG}/aggregate-%j.out" --error="${LOG}/aggregate-%j.err" --export=ALL,RUN_ID="${RUN_ID}" "${SLURM}/aggregate_all1620.sbatch")
printf 'transform=%s\nvalidate=%s\nwllm=%s\nsweet=%s\nsynthid=%s\naggregate=%s\n' "${transform}" "${validate}" "${wllm}" "${sweet}" "${synthid}" "${aggregate}"

#!/bin/bash
set -euo pipefail

RUN_ID="${RUN_ID:?RUN_ID is required}"
RULE_PROFILE="${RULE_PROFILE:?RULE_PROFILE is required}"
ROOT="${HOME}/project/CodeWM/rebuttal_arr/new_prototype"
RUN_ROOT="${ROOT}/data/watermark_attack/${RUN_ID}"
MANIFEST="${RUN_ROOT}/manifest.json"
PYTHON="${HOME}/conda/envs/watermarking/bin/python"

case "${RUN_ID}" in
  rw100-useful-v3-full|rw100-useful-v3-no-advanced) ;;
  *) echo "refusing unapproved run id: ${RUN_ID}" >&2; exit 1 ;;
esac
if [[ ! -f "${MANIFEST}" ]]; then
  echo "missing manifest: ${MANIFEST}" >&2
  exit 1
fi
if [[ -e "${RUN_ROOT}/transforms" || -e "${RUN_ROOT}/scores" || -e "${RUN_ROOT}/summary.json" ]]; then
  echo "refusing to overwrite existing outputs under ${RUN_ROOT}" >&2
  exit 1
fi

"${PYTHON}" -c '
import json, sys
path, expected_run, expected_profile = sys.argv[1:]
manifest = json.load(open(path, encoding="utf-8"))
assert manifest["run_id"] == expected_run, manifest["run_id"]
assert manifest["walk"]["rule_profile"] == expected_profile, manifest["walk"]
assert manifest["counts"]["configs"] == 43, manifest["counts"]
assert manifest["counts"]["walks"] == 11118, manifest["counts"]
assert manifest["counts"]["transform_shards"] == 472, manifest["counts"]
' "${MANIFEST}" "${RUN_ID}" "${RULE_PROFILE}"

RUN_ID="${RUN_ID}" bash "${ROOT}/experiment/slurm/submit.sh"

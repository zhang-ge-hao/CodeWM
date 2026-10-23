#!/usr/bin/env bash
#SBATCH --job-name=pymin-post
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=data/ablation-mini-swe-agent-vllm/python-minifier-slurm-%j.out
#SBATCH --error=data/ablation-mini-swe-agent-vllm/python-minifier-slurm-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?}"
: "${PI_WORKDIR:?PI_WORKDIR must point to the shared work directory}"
: "${SOURCE_DIR:?Set the completed experiment source directory}"
: "${OUTPUT_DIR:?Set the post-obfuscation output directory}"
: "${RUN_ID:?Set the official evaluation run ID}"

ENV_ROOT="${CONDA_ENV_ROOT:-${PI_WORKDIR}/conda/envs}"
QWEN_PYTHON="${ENV_ROOT}/qwen36/bin/python"
SWEBENCH_PYTHON="${ENV_ROOT}/swebench/bin/python"
PYMINIFY="${PYMINIFY_BIN:-${HOME}/conda/envs/watermarking/bin/pyminify}"
REPO_CACHE="${REPO_CACHE:-rebuttal_arr/swe_bench/data/ablation-mini-swe-agent-vllm/repo-cache}"

for executable in "${QWEN_PYTHON}" "${SWEBENCH_PYTHON}" "${PYMINIFY}"; do
    test -x "${executable}"
done
test -d "${SOURCE_DIR}"
mkdir -p "${OUTPUT_DIR}" "${REPO_CACHE}"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${PI_WORKDIR}/huggingface}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export MODAL_BUILD_VALIDATION=ignore
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="${PWD}:${PWD}/rebuttal_arr/swe_bench${PYTHONPATH:+:${PYTHONPATH}}"

args=(
    --source-dir "${SOURCE_DIR}"
    --data-dir "${OUTPUT_DIR}"
    --repo-cache "${REPO_CACHE}"
    --pyminify "${PYMINIFY}"
    --swebench-python "${SWEBENCH_PYTHON}"
    --run-id "${RUN_ID}"
    --transform-workers "${TRANSFORM_WORKERS:-4}"
    --eval-workers "${EVAL_WORKERS:-4}"
    --modal-eval-timeout 1800
)

if [[ -n "${INSTANCE_IDS:-}" ]]; then
    # Slurm's --export syntax reserves commas between environment variables.
    # Use a colon-delimited instance list so multi-case submissions survive.
    IFS=':' read -r -a ids <<<"${INSTANCE_IDS}"
    for instance_id in "${ids[@]}"; do
        args+=(--instance-id "${instance_id}")
    done
fi
if [[ -n "${LIMIT:-}" ]]; then
    args+=(--limit "${LIMIT}")
fi
if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
    args+=(--prepare-only)
fi
if [[ "${SKIP_DETECTION:-0}" == "1" ]]; then
    args+=(--skip-detection)
fi
if [[ "${NO_REUSE_IDENTICAL_EVALUATIONS:-0}" == "1" ]]; then
    args+=(--no-reuse-identical-evaluations)
fi

"${QWEN_PYTHON}" rebuttal_arr/swe_bench/run_python_minifier_postprocess.py "${args[@]}"

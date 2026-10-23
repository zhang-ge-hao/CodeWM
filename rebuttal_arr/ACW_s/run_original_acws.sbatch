#!/usr/bin/env bash
#SBATCH --job-name=acws-repro
#SBATCH --partition=superpod-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=rebuttal_arr/ACW_s/results/slurm-%j.out
#SBATCH --error=rebuttal_arr/ACW_s/results/slurm-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?}"
ACWS_DIR="${PWD}/rebuttal_arr/ACW_s"
PYTHON_BIN="${ACWS_PYTHON:-${HOME}/conda/envs/watermarking/bin/python}"
RUN_ROOT="${ACWS_RUN_ROOT:-${ACWS_DIR}/results/job-${SLURM_JOB_ID}}"
TASKS="${ACWS_TASKS:-humaneval,mbpp}"

test -x "${PYTHON_BIN}"
mkdir -p "${RUN_ROOT}" "${HF_HOME:-${HOME}/.cache/huggingface}"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_ALLOW_CODE_EVAL=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=1

"${PYTHON_BIN}" "${ACWS_DIR}/smoke_test.py"
nvidia-smi

run_task() {
    local task="$1"
    local entropy_threshold="$2"
    local output_dir="${RUN_ROOT}/${task}"
    mkdir -p "${output_dir}"
    "${PYTHON_BIN}" "${ACWS_DIR}/run_upstream.py" \
        --model infly/OpenCoder-1.5B-Instruct \
        --task_name "${task}" \
        --output_dir "${output_dir}" \
        --batch_size 20 \
        --max_length 2048 \
        --precision bf16 \
        --do_sample true \
        --temperature 0.2 \
        --top_k 0 \
        --top_p 0.95 \
        --n_samples 20 \
        --seed 42 \
        --wm sweetcode \
        --gamma 0.5 \
        --delta 2.0 \
        --entropy_threshold "${entropy_threshold}" \
        --detection_z_threshold 4.0 \
        --postprocess true \
        --allow_code_execution true \
        --use_auth_token false
}

IFS=',' read -r -a task_list <<<"${TASKS}"
for task in "${task_list[@]}"; do
    case "${task}" in
        humaneval) run_task humaneval 1.2 ;;
        mbpp) run_task mbpp 0.3 ;;
        *) echo "Unsupported task: ${task}" >&2; exit 2 ;;
    esac
done

"${PYTHON_BIN}" "${ACWS_DIR}/compare_results.py" \
    --run-root "${RUN_ROOT}" \
    --tasks "${TASKS}"

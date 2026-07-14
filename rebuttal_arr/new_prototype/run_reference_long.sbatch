#!/bin/bash
#SBATCH --job-name=rw-ref-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MALLOC_ARENA_MAX=2
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export HYPOTHESIS_STORAGE_DIRECTORY="$PWD/data/.hypothesis"

/usr/bin/python -m pytest \
    -o cache_dir=data/.pytest_cache \
    -m long \
    -s \
    -q \
    tests/test_reference_corpus_long.py

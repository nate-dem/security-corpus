#!/usr/bin/env bash
#SBATCH --job-name=web-near-dedup
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/near-dedup/%x-%j.log
set -euo pipefail
cd "${NEAR_DEDUP_PROJECT:?}"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "${NEAR_DEDUP_PYTHON:?}" -u -m scripts.near_dedup.run \
  --project "$NEAR_DEDUP_PROJECT" --selected "${NEAR_DEDUP_SELECTED:?}" \
  --output "${NEAR_DEDUP_OUTPUT:?}" --workers 7

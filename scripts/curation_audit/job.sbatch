#!/usr/bin/env bash
#SBATCH --job-name=youtube-audit-resume
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/curation/%x-%j.log
set -euo pipefail
cd "${SINGLE_AUDIT_PROJECT:?}"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec "${SINGLE_AUDIT_PYTHON:?}" -u -m scripts.curation_audit.run \
  --plan "${SINGLE_AUDIT_PLAN:?}" --output "${SINGLE_AUDIT_OUTPUT:?}"

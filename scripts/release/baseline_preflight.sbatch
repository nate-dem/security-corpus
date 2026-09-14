#!/usr/bin/env bash
# Recount the retained baseline and inventory release requirements, without rescoring.
#SBATCH --job-name=baseline-preflight
#SBATCH --account=marlowe-m000091
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/release/%x-%j.log
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/scratch/m000091/natedem/security-corpus}"
RELEASE_PYTHON="${RELEASE_PYTHON:-${PROJECT_DIR}/scripts/web_corpora/.venv/bin/python}"
export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "${PROJECT_DIR}"
exec "${RELEASE_PYTHON}" -u -m scripts.release.baseline_preflight \
  --project-root "${PROJECT_DIR}" \
  --output-dir "/scratch/m000091/${USER}/release-review/baseline-${SLURM_JOB_ID}" \
  --workers "${SLURM_CPUS_PER_TASK:-4}"

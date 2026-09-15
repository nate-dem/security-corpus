#!/usr/bin/env bash
#SBATCH --job-name=corpus-production-audit
#SBATCH --account=marlowe-m000091
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/curation/%x-%j.log
set -euo pipefail
cd "${PROJECT_DIR:-/scratch/m000091/natedem/security-corpus}"
export PYTHONPATH="${PWD}/src:${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
exec "${CURATION_CPU_PYTHON:-${PWD}/scripts/web_corpora/.venv/bin/python}" \
  -u -m scripts.curation.production_audit \
  --plan "${CURATION_PRODUCTION_PLAN:?Production plan is required}"

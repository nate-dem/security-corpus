#!/usr/bin/env bash
#SBATCH --job-name=corpus-single-audit
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/curation/%x-%j.log
set -euo pipefail
cd "${PROJECT_DIR:?Set PROJECT_DIR to the repository root}"
export PYTHONPATH="${PWD}/src:${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${CURATION_CPU_PYTHON:-scripts/web_corpora/.venv/bin/python}" -u \
  -m scripts.curation_single.audit --plan "${SINGLE_PLAN:?}"

#!/usr/bin/env bash
#SBATCH --job-name=web-cpu-index
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/web-filter/%x-%j.log
set -euo pipefail
cd "${PROJECT_DIR:?Set PROJECT_DIR to the repository root}"
export PYTHONPATH="${PWD}/src:${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
exec "${WEB_FILTER_PYTHON:-scripts/web_corpora/.venv/bin/python}" -u \
  -m scripts.web_filter.run index --plan "${WEB_FILTER_PLAN:?}"

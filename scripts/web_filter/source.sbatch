#!/usr/bin/env bash
#SBATCH --job-name=web-cpu-filter
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/web-filter/%x-%A_%a.log
set -euo pipefail
cd "${PROJECT_DIR:?Set PROJECT_DIR to the repository root}"
export PYTHONPATH="${PWD}/src:${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
WEB_FILTER_SOURCES=(primus-fineweb redsage-cfw)
exec "${WEB_FILTER_PYTHON:-scripts/web_corpora/.venv/bin/python}" -u \
  -m scripts.web_filter.run source --plan "${WEB_FILTER_PLAN:?}" \
  --source "${WEB_FILTER_SOURCES[${SLURM_ARRAY_TASK_ID:?}]}" --workers 7

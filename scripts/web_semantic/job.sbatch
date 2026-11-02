#!/usr/bin/env bash
#SBATCH --job-name=web-semantic
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/web-semantic/%x-%j.log
set -euo pipefail
cd "${PROJECT_DIR:?Set PROJECT_DIR to the repository root}"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WEB_SEMANTIC_PYTHON="${WEB_SEMANTIC_PYTHON:-$PWD/scripts/web_semantic/.venv/bin/python}"
"$WEB_SEMANTIC_PYTHON" -u -m scripts.web_semantic.run score \
  --plan "${WEB_SEMANTIC_PLAN:?}" --workers 7
if [[ -n "${WEB_SEMANTIC_MIN_SCORE:-}" ]]; then
  "$WEB_SEMANTIC_PYTHON" -u -m scripts.web_semantic.run select \
    --plan "$WEB_SEMANTIC_PLAN" --min-score "$WEB_SEMANTIC_MIN_SCORE" \
    --output "${WEB_SEMANTIC_SELECTION:?}"
fi

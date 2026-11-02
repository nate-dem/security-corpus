#!/usr/bin/env bash
#SBATCH --job-name=youtube-full-finalize
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/youtube-resume/%x-%j.log
set -euo pipefail
cd "${PROJECT_DIR:?}"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"${CURATION_CPU_PYTHON:?}" -u -m scripts.youtube_resume.full_audit \
  --plan "${SINGLE_PLAN:?}" --manifest "${YT_RESUME_MANIFEST:?}" \
  --previous "${YT_RESUME_PREVIOUS:?}" --output "${YT_RESUME_OUTPUT:?}"
exec "${YT_RESUME_DEDUP_PYTHON:?}" -u -m scripts.youtube_finalize.finalize \
  --output "$YT_RESUME_OUTPUT" --web "${YT_RESUME_WEB:?}" --workers 7

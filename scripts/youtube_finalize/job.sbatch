#!/usr/bin/env bash
#SBATCH --job-name=youtube-finalize
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/youtube-finalize/%x-%j.log
set -euo pipefail
cd "${YT_FINISH_PROJECT:?}"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"${YT_FINISH_AUDIT_PYTHON:?}" -u -m scripts.youtube_cpu_io.run \
  --threads "${YT_FINISH_HASH_THREADS:-8}" pipeline \
  --manifest "${YT_FINISH_MANIFEST:?}" --output "${YT_FINISH_OUTPUT:?}"
exec "${YT_FINISH_PYTHON:?}" -u -m scripts.youtube_cpu_io.run \
  --threads "${YT_FINISH_HASH_THREADS:-8}" finalize \
  --output "$YT_FINISH_OUTPUT" --web "${YT_FINISH_WEB:?}" --workers 7

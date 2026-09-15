#!/usr/bin/env bash
#SBATCH --job-name=curation-inputs
#SBATCH --account=marlowe-m000091
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/curation/%x-%j.log
set -euo pipefail
cd "${PROJECT_DIR:-/scratch/m000091/natedem/security-corpus}"
export PYTHONPATH="${PWD}/src:${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
bash scripts/youtube_filter/run.sh --check-cpu
exec "${YOUTUBE_FILTER_CPU_PYTHON:-${PWD}/scripts/web_corpora/.venv/bin/python}" -u -m scripts.curation.prepare \
  --youtube "${YOUTUBE_DATA_DIR:-/scratch/m000091/${USER}/youtube-transcripts}/english-candidates-v1" \
  --web-root "${WEB_DATA_DIR:-/scratch/m000091/${USER}/web-corpora}" \
  --output-dir "${CURATION_PREPARED_DIR:-/scratch/m000091/${USER}/curation/inputs-v1}" \
  --workers "${SLURM_CPUS_PER_TASK:-4}"

#!/usr/bin/env bash
set -euo pipefail
CURATION_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CURATION_PROJECT"
export PYTHONPATH="${CURATION_PROJECT}/src:${CURATION_PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
CURATION_CHECK_PYTHON="${CURATION_CPU_PYTHON:-${CURATION_PROJECT}/scripts/web_corpora/.venv/bin/python}"
"$CURATION_CHECK_PYTHON" -m scripts.curation.first_batch --project "$CURATION_PROJECT" \
  --input-dir "${CURATION_PREPARED:-/scratch/m000091/${USER}/curation/inputs-v1}" \
  --output-dir "${CURATION_FIRST_BATCH_OUTPUT:-/scratch/m000091/${USER}/curation/first-batch-v1}" --check-env
mkdir -p logs/curation
CURATION_SETUP_JOB=$(sbatch --parsable scripts/curation/setup_next.sbatch)
echo "Setup job: $CURATION_SETUP_JOB"
sbatch --dependency="afterok:${CURATION_SETUP_JOB%%;*}" --kill-on-invalid-dep=yes \
  scripts/curation/first_batch.sbatch

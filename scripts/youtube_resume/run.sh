#!/usr/bin/env bash
# Resume unfinished scoring after the existing CPU finalization completes.
set -euo pipefail
export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$PWD/scripts/curation/.venv-next/bin:$PATH"
export CURATION_CPU_PYTHON="${CURATION_CPU_PYTHON:-$PWD/scripts/web_corpora/.venv/bin/python}"
export SINGLE_PLAN="${SINGLE_PLAN:-${CYBEROPEN_WORK_DIR:-${PWD}/data}/curation/single-pass-v1/plan.json}"
export YT_RESUME_MANIFEST="${YT_RESUME_MANIFEST:-${CYBEROPEN_WORK_DIR:-${PWD}/data}/curation/modal-youtube-v1/manifest.json}"
export YT_RESUME_PREVIOUS="${YT_RESUME_PREVIOUS:-${CYBEROPEN_WORK_DIR:-${PWD}/data}/curation/youtube-finalized-v1}"
export YT_RESUME_OUTPUT="${YT_RESUME_OUTPUT:-${CYBEROPEN_WORK_DIR:-${PWD}/data}/curation/youtube-finalized-full-v1}"
export YT_RESUME_WEB="${YT_RESUME_WEB:-${CYBEROPEN_WORK_DIR:-${PWD}/data}/curation/web-near-dedup-v1}"
export YT_RESUME_DEDUP_PYTHON="${YT_RESUME_DEDUP_PYTHON:-$PWD/scripts/web_semantic/.venv/bin/python}"
# Bound resumed GPU allocations; override through SBATCH_TIMELIMIT.
# Explicit sbatch --time on the final CPU job overrides this GPU setting.
export SBATCH_TIMELIMIT="${SBATCH_TIMELIMIT:-01:00:00}"
mkdir -p logs/curation logs/youtube-resume
exec "$CURATION_CPU_PYTHON" -u -m scripts.youtube_resume.run \
  --plan "$SINGLE_PLAN" --manifest "$YT_RESUME_MANIFEST" \
  --previous "$YT_RESUME_PREVIOUS" --output "$YT_RESUME_OUTPUT" \
  --user "$USER" "$@"

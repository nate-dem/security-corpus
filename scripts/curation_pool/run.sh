#!/usr/bin/env bash
# Keep this lightweight controller in tmux while it submits successive waves.
set -euo pipefail
export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${PROJECT_DIR}/scripts/curation/.venv-next/bin:${PATH}"
export CURATION_CPU_PYTHON="${CURATION_CPU_PYTHON:-${PROJECT_DIR}/scripts/web_corpora/.venv/bin/python}"
export SINGLE_PLAN="${CURATION_SINGLE_OUTPUT:-${CYBEROPEN_WORK_DIR:-${PWD}/data}/curation/single-pass-v1}/plan.json"
"$CURATION_CPU_PYTHON" -m scripts.curation_single.run plan \
  --input-dir "${CURATION_PREPARED:-${CYBEROPEN_WORK_DIR:-${PWD}/data}/curation/inputs-v1}" \
  --output-dir "$(dirname "$SINGLE_PLAN")"
mkdir -p logs/curation
exec "$CURATION_CPU_PYTHON" -u -m scripts.curation_pool.run run \
  --plan "$SINGLE_PLAN" --user "$USER" "$@"

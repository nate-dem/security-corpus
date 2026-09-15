#!/usr/bin/env bash
# Freeze a bounded production plan, submit unfinished slots, then audit even if
# a worker failed so partial progress and unresolved slots remain inspectable.
set -euo pipefail
CURATION_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CURATION_PROJECT"
if [[ $# -ne 0 ]]; then echo 'Usage: submit_production.sh' >&2; exit 2; fi
export PROJECT_DIR="$CURATION_PROJECT"
export PYTHONPATH="${CURATION_PROJECT}/src:${CURATION_PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
export CURATION_CPU_PYTHON="${CURATION_CPU_PYTHON:-${CURATION_PROJECT}/scripts/web_corpora/.venv/bin/python}"
CURATION_OUTPUT="${CURATION_PRODUCTION_OUTPUT:-/scratch/m000091/${USER}/curation/production-v1}"
export CURATION_PRODUCTION_PLAN="${CURATION_OUTPUT}/plan.json"
export PATH="${CURATION_PROJECT}/scripts/curation/.venv-next/bin:${PATH}"
"${CURATION_PROJECT}/scripts/curation/.venv-next/bin/python" - <<'PY'
import importlib.metadata as metadata
import json
from pathlib import Path
from scripts.youtube.profile import _sha256
from scripts.curation.cuda_preflight import cccl_include, check_ninja
root = Path('scripts/curation')
ready = json.loads((root/'.venv-next/models-ready.json').read_text())
assert ready['complete'] and ready['models_file_sha256'] == _sha256(root/'next_models.json')
assert ready['requirements_sha256'] == _sha256(root/'requirements-next.txt')
for package, version in ready['versions'].items():
    assert metadata.version(package) == version, f'Installed version changed: {package}'
cccl_include()
print('Runtime ready:', check_ninja(), flush=True)
PY
"$CURATION_CPU_PYTHON" -m scripts.curation.production plan \
  --input-dir "${CURATION_PREPARED:-/scratch/m000091/${USER}/curation/inputs-v1}" \
  --output-dir "$CURATION_OUTPUT"
# Prevent accidental duplicate submissions while an earlier worker/audit is active.
if [[ -f "$CURATION_OUTPUT/submitted-jobs.txt" ]]; then
  while IFS= read -r CURATION_OLD_JOB; do
    if [[ -n "$CURATION_OLD_JOB" ]] && [[ -n "$(squeue -h -j "$CURATION_OLD_JOB" -o '%i' 2>/dev/null || true)" ]]; then
      echo "Production job $CURATION_OLD_JOB is still queued/running; no duplicate submitted." >&2
      exit 0
    fi
  done < "$CURATION_OUTPUT/submitted-jobs.txt"
fi
CURATION_PENDING="$("$CURATION_CPU_PYTHON" -m scripts.curation.production pending --plan "$CURATION_PRODUCTION_PLAN")"
mkdir -p logs/curation
CURATION_DEPENDENCY=()
if [[ -n "$CURATION_PENDING" ]]; then
  CURATION_GPU_JOB="$(sbatch --parsable --array="${CURATION_PENDING}%2" scripts/curation/production.sbatch)"
  CURATION_GPU_JOB="${CURATION_GPU_JOB%%;*}"
  printf '%s\n' "$CURATION_GPU_JOB" > "$CURATION_OUTPUT/submitted-jobs.txt"
  CURATION_DEPENDENCY=(--dependency="afterany:${CURATION_GPU_JOB}")
  echo "Submitted GPU array $CURATION_GPU_JOB (slots $CURATION_PENDING; at most two GPUs concurrently)."
fi
CURATION_AUDIT_JOB="$(sbatch --parsable "${CURATION_DEPENDENCY[@]}" scripts/curation/production_audit.sbatch)"
CURATION_AUDIT_JOB="${CURATION_AUDIT_JOB%%;*}"
printf '%s\n' "$CURATION_AUDIT_JOB" >> "$CURATION_OUTPUT/submitted-jobs.txt"
echo "Submitted CPU audit $CURATION_AUDIT_JOB. Reports: $CURATION_OUTPUT/review/"

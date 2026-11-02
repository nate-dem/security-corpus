#!/usr/bin/env bash
# Submit only the requested source; total concurrency is capped at eight GPUs.
set -euo pipefail
export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"
SINGLE_SOURCE="${1:-youtube-commons}"
case "$SINGLE_SOURCE" in benchmark|youtube-commons|primus-fineweb|redsage-cfw|all) ;; *) echo 'Usage: submit.sh benchmark|youtube-commons|primus-fineweb|redsage-cfw|all' >&2; exit 2 ;; esac
[[ $# -le 1 ]] || exit 2
export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${PROJECT_DIR}/scripts/curation/.venv-next/bin:${PATH}"
export CURATION_CPU_PYTHON="${CURATION_CPU_PYTHON:-${PROJECT_DIR}/scripts/web_corpora/.venv/bin/python}"
SINGLE_OUTPUT="${CURATION_SINGLE_OUTPUT:-${CYBEROPEN_WORK_DIR:-${PWD}/data}/curation/single-pass-v1}"
export SINGLE_PLAN="${SINGLE_OUTPUT}/plan.json"
# Check cached runtime without loading weights or needing a GPU on the login node.
scripts/curation/.venv-next/bin/python - <<'PY'
import importlib.metadata as metadata
import json
from pathlib import Path
from scripts.youtube.profile import _sha256
from scripts.curation.cuda_preflight import cccl_include, check_ninja
root = Path('scripts/curation')
ready = json.loads((root/'.venv-next/models-ready.json').read_text())
assert ready['complete']
assert ready['models_file_sha256'] == _sha256(root/'next_models.json')
assert ready['requirements_sha256'] == _sha256(root/'requirements-next.txt')
for package, version in ready['versions'].items():
    assert metadata.version(package) == version, f'Runtime version changed: {package}'
cccl_include()
print('Runtime ready:', check_ninja(), flush=True)
PY
"$CURATION_CPU_PYTHON" -m scripts.curation_single.run plan \
  --input-dir "${CURATION_PREPARED:-${CYBEROPEN_WORK_DIR:-${PWD}/data}/curation/inputs-v1}" --output-dir "$SINGLE_OUTPUT"
# mkdir is an atomic submission lock, distinct from the scorer's file lock.
if ! mkdir "$SINGLE_OUTPUT/.submit-lock" 2>/dev/null; then
  echo 'Another submission is in progress; inspect .submit-lock if a previous submit was interrupted.' >&2; exit 1
fi
trap 'rmdir "$SINGLE_OUTPUT/.submit-lock"' EXIT
if [[ -f "$SINGLE_OUTPUT/submitted-jobs.txt" ]]; then
  while IFS= read -r SINGLE_OLD_JOB; do
    [[ -n "$SINGLE_OLD_JOB" ]] || continue
    # Fail closed if Slurm cannot answer; do not risk duplicate scoring.
    SINGLE_QUEUED="$(squeue -h -u "$USER" -o '%i' | cut -d_ -f1 | sort -u)"
    if [[ $'\n'"$SINGLE_QUEUED"$'\n' == *$'\n'"$SINGLE_OLD_JOB"$'\n'* ]]; then
      echo "Single-pass job $SINGLE_OLD_JOB is still queued/running; no duplicate submitted."; exit 0
    fi
  done < "$SINGLE_OUTPUT/submitted-jobs.txt"
fi
export SINGLE_WORK_LIST="$SINGLE_OUTPUT/work-$(date -u +%Y%m%dT%H%M%S)-$$.json"
SINGLE_JOBS="$("$CURATION_CPU_PYTHON" -m scripts.curation_single.run pending --plan "$SINGLE_PLAN" --source "$SINGLE_SOURCE" --work-list "$SINGLE_WORK_LIST")"
mkdir -p logs/curation
SINGLE_DEPENDENCY=()
if (( SINGLE_JOBS > 0 )); then
  SINGLE_TIME=04:00:00
  [[ "$SINGLE_SOURCE" != benchmark ]] || SINGLE_TIME=01:00:00
  SINGLE_GPU_JOB="$(sbatch --parsable --time="$SINGLE_TIME" --array="0-$((SINGLE_JOBS-1))%8" scripts/curation_single/gpu.sbatch)"
  SINGLE_GPU_JOB="${SINGLE_GPU_JOB%%;*}"
  printf '%s\n' "$SINGLE_GPU_JOB" >> "$SINGLE_OUTPUT/submitted-jobs.txt"
  SINGLE_DEPENDENCY=(--dependency="afterany:${SINGLE_GPU_JOB}")
  echo "Submitted $SINGLE_JOBS $SINGLE_SOURCE workers as array $SINGLE_GPU_JOB, at most eight GPUs."
  echo "To stop pending/running scoring at your deadline, run: scancel $SINGLE_GPU_JOB"
fi
SINGLE_AUDIT_JOB="$(sbatch --parsable "${SINGLE_DEPENDENCY[@]}" scripts/curation_single/audit.sbatch)"
SINGLE_AUDIT_JOB="${SINGLE_AUDIT_JOB%%;*}"
printf '%s\n' "$SINGLE_AUDIT_JOB" >> "$SINGLE_OUTPUT/submitted-jobs.txt"
echo "Submitted CPU audit $SINGLE_AUDIT_JOB; reports: $SINGLE_OUTPUT/review/"

#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 0 ]] || { echo 'Usage: bash scripts/curation_audit/submit.sh' >&2; exit 2; }
export SINGLE_AUDIT_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SINGLE_AUDIT_PROJECT"
export SINGLE_AUDIT_PYTHON="${SINGLE_AUDIT_PYTHON:-$PWD/scripts/web_corpora/.venv/bin/python}"
export SINGLE_AUDIT_PLAN="${SINGLE_AUDIT_PLAN:-${CYBEROPEN_WORK_DIR:-${PWD}/data}/curation/single-pass-v1/plan.json}"
export SINGLE_AUDIT_OUTPUT="${SINGLE_AUDIT_OUTPUT:-${CYBEROPEN_WORK_DIR:-${PWD}/data}/curation/single-pass-audit-v1}"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
"$SINGLE_AUDIT_PYTHON" -c 'from scripts.curation_audit import run; print("CPU audit environment OK; no model loaded")'
[[ -f "$SINGLE_AUDIT_PLAN" ]] || { echo "Missing scoring plan: $SINGLE_AUDIT_PLAN" >&2; exit 1; }
mkdir -p "$SINGLE_AUDIT_OUTPUT" logs/curation
if ! mkdir "$SINGLE_AUDIT_OUTPUT/.submit-lock" 2>/dev/null; then
  echo 'Submission lock exists; inspect before retrying.' >&2; exit 1
fi
trap 'rmdir "$SINGLE_AUDIT_OUTPUT/.submit-lock"' EXIT
SINGLE_AUDIT_QUEUED="$(squeue -h -u "$USER" -o '%i' | cut -d_ -f1 | sort -u)"
if [[ -f "$SINGLE_AUDIT_OUTPUT/submitted-jobs.txt" ]]; then
  while IFS= read -r SINGLE_AUDIT_JOB; do
    if [[ $'\n'"$SINGLE_AUDIT_QUEUED"$'\n' == *$'\n'"$SINGLE_AUDIT_JOB"$'\n'* ]]; then
      echo "Job $SINGLE_AUDIT_JOB is active; no duplicate submitted."; exit 0
    fi
  done < "$SINGLE_AUDIT_OUTPUT/submitted-jobs.txt"
fi
if [[ -f "$SINGLE_AUDIT_OUTPUT/submission-in-progress" ]]; then
  echo 'Earlier submission outcome uncertain; inspect queue before retrying.' >&2; exit 1
fi
date -u > "$SINGLE_AUDIT_OUTPUT/submission-in-progress"
if ! SINGLE_AUDIT_JOB="$(sbatch --parsable --time="${SINGLE_AUDIT_TIME:-04:00:00}" scripts/curation_audit/job.sbatch)"; then
  rm "$SINGLE_AUDIT_OUTPUT/submission-in-progress"
  exit 1
fi
SINGLE_AUDIT_JOB="${SINGLE_AUDIT_JOB%%;*}"
[[ "$SINGLE_AUDIT_JOB" =~ ^[0-9]+$ ]] || { echo 'Ambiguous sbatch reply; inspect queue.' >&2; exit 1; }
printf '%s\n' "$SINGLE_AUDIT_JOB" >> "$SINGLE_AUDIT_OUTPUT/submitted-jobs.txt"
rm "$SINGLE_AUDIT_OUTPUT/submission-in-progress"
echo "Submitted resumable CPU audit $SINGLE_AUDIT_JOB (4 CPUs, 16 GB, no GPUs)."
echo "Reports: $SINGLE_AUDIT_OUTPUT/review/"

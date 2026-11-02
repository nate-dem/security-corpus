#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 0 ]] || { echo 'Usage: bash scripts/near_dedup/submit.sh' >&2; exit 2; }
export NEAR_DEDUP_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$NEAR_DEDUP_PROJECT"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export NEAR_DEDUP_PYTHON="${NEAR_DEDUP_PYTHON:-$PWD/scripts/web_semantic/.venv/bin/python}"
export NEAR_DEDUP_SELECTED="${NEAR_DEDUP_SELECTED:-${CYBEROPEN_WORK_DIR:-${PWD}/data}/curation/web-semantic-selected-v1}"
export NEAR_DEDUP_OUTPUT="${NEAR_DEDUP_OUTPUT:-${CYBEROPEN_WORK_DIR:-${PWD}/data}/curation/web-near-dedup-v1}"
"$NEAR_DEDUP_PYTHON" -m scripts.near_dedup.run --check-env
"$NEAR_DEDUP_PYTHON" -m scripts.near_dedup.run --prepare \
  --project "$NEAR_DEDUP_PROJECT" --selected "$NEAR_DEDUP_SELECTED" --output "$NEAR_DEDUP_OUTPUT"
if ! mkdir "$NEAR_DEDUP_OUTPUT/.submit-lock" 2>/dev/null; then
  echo 'Submission lock exists; inspect before retrying.' >&2; exit 1
fi
trap 'rmdir "$NEAR_DEDUP_OUTPUT/.submit-lock"' EXIT
NEAR_DEDUP_QUEUED="$(squeue -h -u "$USER" -o '%i' | cut -d_ -f1 | sort -u)"
if [[ -f "$NEAR_DEDUP_OUTPUT/submitted-jobs.txt" ]]; then
  while IFS= read -r NEAR_DEDUP_JOB; do
    if [[ $'\n'"$NEAR_DEDUP_QUEUED"$'\n' == *$'\n'"$NEAR_DEDUP_JOB"$'\n'* ]]; then
      echo "Job $NEAR_DEDUP_JOB is active; no duplicate submitted."; exit 0
    fi
  done < "$NEAR_DEDUP_OUTPUT/submitted-jobs.txt"
fi
if [[ -f "$NEAR_DEDUP_OUTPUT/submission-in-progress" ]]; then
  echo 'Earlier submission outcome uncertain; inspect squeue and submission-in-progress.' >&2; exit 1
fi
mkdir -p logs/near-dedup
date -u > "$NEAR_DEDUP_OUTPUT/submission-in-progress"
if ! NEAR_DEDUP_JOB="$(sbatch --parsable --time="${NEAR_DEDUP_TIME:-04:00:00}" scripts/near_dedup/job.sbatch)"; then
  rm "$NEAR_DEDUP_OUTPUT/submission-in-progress"
  exit 1
fi
NEAR_DEDUP_JOB="${NEAR_DEDUP_JOB%%;*}"
[[ "$NEAR_DEDUP_JOB" =~ ^[0-9]+$ ]] || { echo 'Ambiguous sbatch reply; inspect squeue before retrying.' >&2; exit 1; }
printf '%s\n' "$NEAR_DEDUP_JOB" >> "$NEAR_DEDUP_OUTPUT/submitted-jobs.txt"
rm "$NEAR_DEDUP_OUTPUT/submission-in-progress"
echo "Submitted CPU job $NEAR_DEDUP_JOB (8 CPUs, 32 GB, no GPUs)."
echo "Reports: $NEAR_DEDUP_OUTPUT/review/"

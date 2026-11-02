#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 0 ]] || { echo 'Usage: bash scripts/web_semantic/submit.sh' >&2; exit 2; }
export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export WEB_SEMANTIC_PYTHON="${WEB_SEMANTIC_PYTHON:-$PWD/scripts/web_semantic/.venv/bin/python}"
WEB_SEMANTIC_OUTPUT="${WEB_SEMANTIC_OUTPUT:-${CYBEROPEN_WORK_DIR:-${PWD}/data}/curation/web-semantic-v1}"
export WEB_SEMANTIC_PLAN="$WEB_SEMANTIC_OUTPUT/plan.json"
export WEB_SEMANTIC_SELECTION="${WEB_SEMANTIC_SELECTION:-${CYBEROPEN_WORK_DIR:-${PWD}/data}/curation/web-semantic-selected-v1}"
"$WEB_SEMANTIC_PYTHON" -m scripts.web_semantic.run check-env
"$WEB_SEMANTIC_PYTHON" -m scripts.web_semantic.run plan \
  --web "${WEB_FILTER_OUTPUT:-${CYBEROPEN_WORK_DIR:-${PWD}/data}/curation/web-cpu-v1}" \
  --model "${WEB_SEMANTIC_MODEL:-$PWD/models/web-semantic-v1}" \
  --output "$WEB_SEMANTIC_OUTPUT"
if ! mkdir "$WEB_SEMANTIC_OUTPUT/.submit-lock" 2>/dev/null; then
  echo 'Submission lock exists; inspect before retrying.' >&2; exit 1
fi
trap 'rmdir "$WEB_SEMANTIC_OUTPUT/.submit-lock"' EXIT
WEB_SEMANTIC_QUEUED="$(squeue -h -u "$USER" -o '%i' | cut -d_ -f1 | sort -u)"
if [[ -f "$WEB_SEMANTIC_OUTPUT/submitted-jobs.txt" ]]; then
  while IFS= read -r WEB_SEMANTIC_JOB; do
    if [[ $'\n'"$WEB_SEMANTIC_QUEUED"$'\n' == *$'\n'"$WEB_SEMANTIC_JOB"$'\n'* ]]; then
      echo "Job $WEB_SEMANTIC_JOB is active; no duplicate submitted."; exit 0
    fi
  done < "$WEB_SEMANTIC_OUTPUT/submitted-jobs.txt"
fi
if [[ -f "$WEB_SEMANTIC_OUTPUT/submission-in-progress" ]]; then
  echo 'Earlier submission outcome uncertain. Inspect squeue and submission-in-progress before retrying.' >&2; exit 1
fi
mkdir -p logs/web-semantic
date -u > "$WEB_SEMANTIC_OUTPUT/submission-in-progress"
if ! WEB_SEMANTIC_JOB="$(sbatch --parsable scripts/web_semantic/job.sbatch)"; then
  rm "$WEB_SEMANTIC_OUTPUT/submission-in-progress"
  exit 1
fi
WEB_SEMANTIC_JOB="${WEB_SEMANTIC_JOB%%;*}"
[[ "$WEB_SEMANTIC_JOB" =~ ^[0-9]+$ ]] || { echo 'Ambiguous sbatch reply; inspect squeue before retrying.' >&2; exit 1; }
printf '%s\n' "$WEB_SEMANTIC_JOB" >> "$WEB_SEMANTIC_OUTPUT/submitted-jobs.txt"
rm "$WEB_SEMANTIC_OUTPUT/submission-in-progress"
echo "Submitted CPU job $WEB_SEMANTIC_JOB (8 CPUs, 32 GB, no GPUs)."
echo "Reports: $WEB_SEMANTIC_OUTPUT/review/"

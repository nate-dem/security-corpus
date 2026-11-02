#!/usr/bin/env bash
# No GPU, network downloads, or model inference. Resume verified part checkpoints.
set -euo pipefail
[[ $# -eq 0 ]] || { echo 'Usage: bash scripts/web_filter/submit.sh' >&2; exit 2; }
export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export WEB_FILTER_PYTHON="${WEB_FILTER_PYTHON:-${PROJECT_DIR}/scripts/web_corpora/.venv/bin/python}"
WEB_FILTER_OUTPUT="${WEB_FILTER_OUTPUT:-${CYBEROPEN_WORK_DIR:-${PWD}/data}/curation/web-cpu-v1}"
export WEB_FILTER_PLAN="$WEB_FILTER_OUTPUT/plan.json"
bash scripts/web_corpora/run.sh --check-env
"$WEB_FILTER_PYTHON" -m scripts.web_filter.run plan --project "$PROJECT_DIR" \
  --prepared "${CURATION_PREPARED:-${CYBEROPEN_WORK_DIR:-${PWD}/data}/curation/inputs-v1}" \
  --web-root "${WEB_DATA_DIR:-${CYBEROPEN_WORK_DIR:-${PWD}/data}/web-corpora}" \
  --output "$WEB_FILTER_OUTPUT"
if ! mkdir "$WEB_FILTER_OUTPUT/.submit-lock" 2>/dev/null; then
  echo 'Another CPU submission is in progress; inspect .submit-lock if an earlier submit was interrupted.' >&2; exit 1
fi
trap 'rmdir "$WEB_FILTER_OUTPUT/.submit-lock"' EXIT
if [[ -f "$WEB_FILTER_OUTPUT/submitted-jobs.txt" ]]; then
  WEB_FILTER_QUEUED="$(squeue -h -u "$USER" -o '%i' | cut -d_ -f1 | sort -u)"
  while IFS= read -r WEB_FILTER_OLD_JOB; do
    [[ -n "$WEB_FILTER_OLD_JOB" ]] || continue
    if [[ $'\n'"$WEB_FILTER_QUEUED"$'\n' == *$'\n'"$WEB_FILTER_OLD_JOB"$'\n'* ]]; then
      echo "CPU job $WEB_FILTER_OLD_JOB is queued/running; no duplicate submitted."; exit 0
    fi
  done < "$WEB_FILTER_OUTPUT/submitted-jobs.txt"
fi
mkdir -p logs/web-filter
WEB_FILTER_INDEX="$(sbatch --parsable scripts/web_filter/index.sbatch)"
WEB_FILTER_INDEX="${WEB_FILTER_INDEX%%;*}"
printf '%s\n' "$WEB_FILTER_INDEX" >> "$WEB_FILTER_OUTPUT/submitted-jobs.txt"
WEB_FILTER_JOB="$(sbatch --parsable --array=0-1%2 --dependency="afterok:$WEB_FILTER_INDEX" --kill-on-invalid-dep=yes scripts/web_filter/source.sbatch)"
WEB_FILTER_JOB="${WEB_FILTER_JOB%%;*}"
printf '%s\n' "$WEB_FILTER_JOB" >> "$WEB_FILTER_OUTPUT/submitted-jobs.txt"
echo "Submitted baseline index $WEB_FILTER_INDEX and CPU source array $WEB_FILTER_JOB. No GPUs requested."
echo "Reports: $WEB_FILTER_OUTPUT/{primus-fineweb,redsage-cfw}/review/"

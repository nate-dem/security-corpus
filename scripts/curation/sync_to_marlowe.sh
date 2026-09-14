#!/usr/bin/env bash
# Refresh code plus only the small, assistant-reviewed development inputs.
set -euo pipefail
CURATION_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CURATION_LOCAL_PYTHON="${CURATION_LOCAL_PYTHON:-${CURATION_PROJECT}/venv/bin/python}"
CURATION_LOCAL_INPUT="${CURATION_PROJECT}/reports/curation/development-v2"
CURATION_REMOTE_HOST="${REMOTE_HOST:-natedem@login.marlowe.stanford.edu}"
CURATION_REMOTE_ROOT="${REMOTE_ROOT:-/scratch/m000091/natedem/security-corpus}"
if [[ $# -ne 0 ]]; then echo 'Usage: sync_to_marlowe.sh' >&2; exit 2; fi
PYTHONPATH="${CURATION_PROJECT}/src:${CURATION_PROJECT}" "$CURATION_LOCAL_PYTHON" - "$CURATION_LOCAL_INPUT" <<'PY'
import sys
from pathlib import Path
from scripts.curation.evaluate import evaluate
p=Path(sys.argv[1])
result=evaluate(p/'packet.json',p/'assistant_annotations.json')
print(f"Validated {result['cases']} assistant-reviewed development cases.",flush=True)
PY
bash "${CURATION_PROJECT}/scripts/marlowe/sync_code.sh"
printf -v CURATION_REMOTE_QUOTED '%q' "${CURATION_REMOTE_ROOT}/reports/curation/development-v2"
ssh "$CURATION_REMOTE_HOST" "mkdir -p -- ${CURATION_REMOTE_QUOTED}"
rsync -av "$CURATION_LOCAL_INPUT/packet.json" "$CURATION_LOCAL_INPUT/assistant_annotations.json" \
  "$CURATION_REMOTE_HOST:${CURATION_REMOTE_ROOT}/reports/curation/development-v2/"
echo 'Code and reviewed inputs copied. Ready for scripts/curation/benchmark.sbatch.'

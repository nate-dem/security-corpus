#!/usr/bin/env bash
set -euo pipefail
CURATION_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export REMOTE_HOST="${REMOTE_HOST:-natedem@login-01.marlowe.stanford.edu}"
export REMOTE_ROOT="${REMOTE_ROOT:-/scratch/m000091/natedem/security-corpus}"
CURATION_PYTHON="${CURATION_LOCAL_PYTHON:-${CURATION_PROJECT}/venv/bin/python}"
if [[ $# -ne 0 ]]; then echo 'Usage: sync_first_batch.sh' >&2; exit 2; fi
PYTHONPATH="${CURATION_PROJECT}/src:${CURATION_PROJECT}" "$CURATION_PYTHON" - "$CURATION_PROJECT" <<'PY'
import sys
from pathlib import Path
from scripts.curation.evaluate import evaluate
p=Path(sys.argv[1])/'reports/curation/accepted-audit-v1'
r=evaluate(p/'packet.json',p/'assistant_annotations.json')
print(f"Validated {r['cases']} assistant-reviewed accepted-case controls.")
PY
bash "${CURATION_PROJECT}/scripts/curation/sync_to_marlowe.sh"
printf -v CURATION_DEST '%q' "${REMOTE_ROOT}/reports/curation/accepted-audit-v1"
ssh "$REMOTE_HOST" "mkdir -p -- $CURATION_DEST"
rsync -av "${CURATION_PROJECT}/reports/curation/accepted-audit-v1/packet.json" \
  "${CURATION_PROJECT}/reports/curation/accepted-audit-v1/assistant_annotations.json" \
  "$REMOTE_HOST:${REMOTE_ROOT}/reports/curation/accepted-audit-v1/"
echo 'Copied code and bounded review inputs. Follow scripts/curation/FIRST_BATCH.md.'

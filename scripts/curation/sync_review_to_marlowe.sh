#!/usr/bin/env bash
set -euo pipefail
CURATION_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CURATION_HOST="${REMOTE_HOST:-natedem@login.marlowe.stanford.edu}"
CURATION_ROOT="${REMOTE_ROOT:-/scratch/m000091/natedem/security-corpus}"
CURATION_PYTHON="${CURATION_LOCAL_PYTHON:-${CURATION_PROJECT}/venv/bin/python}"
if [[ $# -ne 0 ]]; then echo 'Usage: sync_review_to_marlowe.sh' >&2; exit 2; fi
PYTHONPATH="${CURATION_PROJECT}/src:${CURATION_PROJECT}" "$CURATION_PYTHON" - "$CURATION_PROJECT" <<'PY'
import sys
from pathlib import Path
from scripts.curation.score import load_packet
p=Path(sys.argv[1])/'reports/curation/additional-v1/packet.json'
r=load_packet(p)
print(f"Validated {len(r['cases'])} additional diagnostic cases.",flush=True)
PY
bash "${CURATION_PROJECT}/scripts/curation/sync_to_marlowe.sh"
printf -v CURATION_DEST '%q' "${CURATION_ROOT}/reports/curation/additional-v1"
ssh "$CURATION_HOST" "mkdir -p -- ${CURATION_DEST}"
rsync -av "${CURATION_PROJECT}/reports/curation/additional-v1/packet.json" \
  "$CURATION_HOST:${CURATION_ROOT}/reports/curation/additional-v1/"
echo 'Review job ready. Follow scripts/curation/GPU_REVIEW.md; leave the running CPU job alone.'

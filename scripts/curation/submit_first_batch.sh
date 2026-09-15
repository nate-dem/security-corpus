#!/usr/bin/env bash
set -euo pipefail
CURATION_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --reuse-setup ) ]]; then
  echo 'Usage: submit_first_batch.sh [--reuse-setup]' >&2; exit 2
fi
cd "$CURATION_PROJECT"
export PYTHONPATH="${CURATION_PROJECT}/src:${CURATION_PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
CURATION_CHECK_PYTHON="${CURATION_CPU_PYTHON:-${CURATION_PROJECT}/scripts/web_corpora/.venv/bin/python}"
"$CURATION_CHECK_PYTHON" -m scripts.curation.first_batch --project "$CURATION_PROJECT" \
  --input-dir "${CURATION_PREPARED:-/scratch/m000091/${USER}/curation/inputs-v1}" \
  --output-dir "${CURATION_FIRST_BATCH_OUTPUT:-/scratch/m000091/${USER}/curation/first-batch-v4}" --check-env
mkdir -p logs/curation
if [[ "${1:-}" == --reuse-setup ]]; then
  export PATH="${CURATION_PROJECT}/scripts/curation/.venv-next/bin:${PATH}"
  scripts/curation/.venv-next/bin/python - <<'PY'
import importlib.metadata as m
import json
from scripts.curation.first_batch import HERE
from scripts.youtube.profile import _sha256
r=json.loads((HERE/'.venv-next/models-ready.json').read_text())
assert r['complete'] and r['models_file_sha256']==_sha256(HERE/'next_models.json')
assert r['requirements_sha256']==_sha256(HERE/'requirements-next.txt')
for package,version in r['versions'].items():
    assert m.version(package)==version, f'Environment changed: {package}'
# Check installed header discovery without a compiler/GPU on the login node.
from scripts.curation.cuda_preflight import cccl_include, check_ninja
print('Existing model setup verified; CCCL headers:', cccl_include()[0])
print('Build tool verified:', check_ninja())
PY
  exec sbatch scripts/curation/first_batch.sbatch
fi
CURATION_SETUP_JOB=$(sbatch --parsable scripts/curation/setup_next.sbatch)
echo "Setup job: $CURATION_SETUP_JOB"
sbatch --dependency="afterok:${CURATION_SETUP_JOB%%;*}" --kill-on-invalid-dep=yes \
  scripts/curation/first_batch.sbatch

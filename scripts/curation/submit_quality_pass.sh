#!/usr/bin/env bash
# Use the completed model setup and first-batch-v4 source packets.
set -euo pipefail
CURATION_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CURATION_PROJECT"
if [[ $# -ne 0 ]]; then echo 'Usage: submit_quality_pass.sh' >&2; exit 2; fi
export PYTHONPATH="${CURATION_PROJECT}/src:${CURATION_PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
CURATION_CPU_PYTHON="${CURATION_CPU_PYTHON:-${CURATION_PROJECT}/scripts/web_corpora/.venv/bin/python}"
"$CURATION_CPU_PYTHON" -m scripts.curation.quality_pass \
  --prior-dir "${CURATION_QUALITY_PRIOR:-/scratch/m000091/${USER}/curation/first-batch-v4/qwen38-27b-fp8}" \
  --output-dir "${CURATION_QUALITY_OUTPUT:-/scratch/m000091/${USER}/curation/quality-pass-v1/qwen38-27b-fp8}" \
  --check-env
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
print('Runtime and Ninja ready:', check_ninja(), flush=True)
PY
mkdir -p logs/curation
exec sbatch scripts/curation/quality_pass.sbatch

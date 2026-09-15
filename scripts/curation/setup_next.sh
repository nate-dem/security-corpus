#!/usr/bin/env bash
set -euo pipefail
if [[ "$(uname -s)" != Linux ]]; then echo 'Run this on Marlowe, not the Mac.' >&2; exit 2; fi
CURATION_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CURATION_PROJECT"
export PYTHONPATH="${CURATION_PROJECT}/src:${CURATION_PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_CACHE="${CURATION_NEXT_MODEL_CACHE:-/scratch/m000091/${USER}/curation-model-cache}"
export PIP_CACHE_DIR="${HF_HUB_CACHE}/pip-cache"
export HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
CURATION_VENV="${CURATION_PROJECT}/scripts/curation/.venv-next"
if [[ ! -f "$CURATION_VENV/models-ready.json" ]]; then
  python3 -m venv "$CURATION_VENV"
  "$CURATION_VENV/bin/python" -m pip install --upgrade pip
  "$CURATION_VENV/bin/python" -m pip install -r scripts/curation/requirements-next.txt
else
  "$CURATION_VENV/bin/python" - <<'PY'
import importlib.metadata as m
import json
from scripts.curation.first_batch import HERE
from scripts.youtube.profile import _sha256
r=json.loads((HERE/'.venv-next/models-ready.json').read_text())
assert r['requirements_sha256']==_sha256(HERE/'requirements-next.txt'), 'Preserve this environment; use a new environment for changed requirements'
for package,version in r['versions'].items():
    assert m.version(package)==version, f'Environment changed: {package}'
PY
fi
"$CURATION_VENV/bin/python" -m pip check
"$CURATION_VENV/bin/python" -m pip freeze > "$CURATION_VENV/installed-packages.txt"
"$CURATION_VENV/bin/python" -u -m scripts.curation.prepare_next_models
echo 'Comparison environment and pinned model cache ready. Existing environments were preserved.'

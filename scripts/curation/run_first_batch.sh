#!/usr/bin/env bash
set -euo pipefail
CURATION_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CURATION_PROJECT"
export PYTHONPATH="${CURATION_PROJECT}/src:${CURATION_PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_CACHE="${CURATION_NEXT_MODEL_CACHE:-/scratch/m000091/${USER}/curation-model-cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
# The optional random sampler needs missing cuRAND headers. Use vLLM's native
# backend; classification remains temperature-zero greedy decoding.
export VLLM_USE_FLASHINFER_SAMPLER=0
CURATION_VENV="${CURATION_PROJECT}/scripts/curation/.venv-next"
export PATH="${CURATION_VENV}/bin:${PATH}"
CURATION_PYTHON="${CURATION_VENV}/bin/python"
CURATION_INPUT="${CURATION_PREPARED:-/scratch/m000091/${USER}/curation/inputs-v1}"
CURATION_OUTPUT="${CURATION_FIRST_BATCH_OUTPUT:-/scratch/m000091/${USER}/curation/first-batch-v4}"
if [[ "${1:-}" == --check-env ]]; then
  exec "$CURATION_PYTHON" -m scripts.curation.first_batch --project "$CURATION_PROJECT" \
    --input-dir "$CURATION_INPUT" --output-dir "$CURATION_OUTPUT" --check-env
fi
if [[ $# -ne 1 || ! "$1" =~ ^[01]$ ]]; then echo 'Usage: run_first_batch.sh --check-env|0|1' >&2; exit 2; fi
"$CURATION_PYTHON" - <<'PY'
import importlib.metadata as m
import json
from pathlib import Path
import torch
from scripts.curation.first_batch import HERE
from scripts.youtube.profile import _sha256
ready=json.loads((HERE/'.venv-next/models-ready.json').read_text())
assert ready['complete'] and ready['models_file_sha256']==_sha256(HERE/'next_models.json')
for package,version in ready['versions'].items():
    assert m.version(package)==version, f'Environment changed: {package}; preserve this run and use a new output directory'
assert ready['requirements_sha256']==_sha256(HERE/'requirements-next.txt')
assert m.version('vllm') == '0.29.0'
assert torch.cuda.is_available(), 'A compatible GPU allocation is required'
print('GPU:',torch.cuda.get_device_name(0),'CUDA build:',torch.version.cuda,flush=True)
PY
exec "$CURATION_PYTHON" -u -m scripts.curation.first_batch --project "$CURATION_PROJECT" \
  --input-dir "$CURATION_INPUT" --output-dir "$CURATION_OUTPUT" --model-index "$1"

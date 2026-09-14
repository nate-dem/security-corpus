#!/usr/bin/env bash
set -euo pipefail
FILTER_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FILTER_PROJECT_ROOT="$(cd -- "${FILTER_SCRIPT_DIR}/../.." && pwd)"
YOUTUBE_DATA_DIR="${YOUTUBE_DATA_DIR:-/scratch/m000091/${USER}/youtube-transcripts}"
YOUTUBE_PILOT_DIR="${YOUTUBE_PILOT_DIR:-${YOUTUBE_DATA_DIR}/pilot-v1}"
YOUTUBE_SCORE_DIR="${YOUTUBE_SCORE_DIR:-${YOUTUBE_DATA_DIR}/pilot-qwen8b-v1}"
YOUTUBE_FILTER_CPU_PYTHON="${YOUTUBE_FILTER_CPU_PYTHON:-${FILTER_PROJECT_ROOT}/scripts/web_corpora/.venv/bin/python}"
YOUTUBE_FILTER_GPU_PYTHON="${YOUTUBE_FILTER_GPU_PYTHON:-${FILTER_SCRIPT_DIR}/.venv/bin/python}"
export PYTHONPATH="${FILTER_PROJECT_ROOT}/src:${FILTER_PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_CACHE="${YOUTUBE_MODEL_CACHE:-${YOUTUBE_DATA_DIR}/model-cache}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
FILTER_PHASE="${1:---check-cpu}"
if [[ $# -gt 0 ]]; then shift; fi
case "${FILTER_PHASE}" in
  --check-cpu|prepare) FILTER_PYTHON="${YOUTUBE_FILTER_CPU_PYTHON}"; FILTER_ENV=cpu ;;
  --check-gpu-env|score|dry-run|repair-evidence) FILTER_PYTHON="${YOUTUBE_FILTER_GPU_PYTHON}"; FILTER_ENV=gpu ;;
  *) echo "Usage: $0 {--check-cpu|--check-gpu-env|prepare|score|dry-run|repair-evidence} [arguments]" >&2; exit 2 ;;
esac
if ! command -v "${FILTER_PYTHON}" >/dev/null 2>&1; then
  echo "Missing interpreter: ${FILTER_PYTHON}; follow scripts/youtube_filter/README.md setup." >&2
  exit 1
fi
"${FILTER_PYTHON}" - "${FILTER_ENV}" <<'PY'
import importlib.metadata
import sys

try:
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10+ required")
    from ingest.utils import compute_token_count
    assert compute_token_count("Offline tokenizer check.") > 0
    expected = {"duckdb": "1.5.2", "pyarrow": "23.0.1", "tiktoken": "0.12.0"}
    if sys.argv[1] == "gpu":
        expected.update({"vllm": "0.13.0", "transformers": "4.57.3", "huggingface-hub": "0.36.0", "jinja2": "3.1.6"})
    for name, version in expected.items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f"Expected {name}=={version}")
    import duckdb
    import pyarrow.parquet
    if sys.argv[1] == "gpu":
        import torch
        from transformers import AutoTokenizer
        from vllm.sampling_params import StructuredOutputsParams
except Exception as exc:
    print(f"Environment check failed in {sys.executable}: {exc}", file=sys.stderr)
    print("Install the matching requirements file before submitting. See scripts/youtube_filter/README.md.", file=sys.stderr)
    raise SystemExit(1)
print(f"{sys.argv[1].upper()} environment OK: {sys.executable}", flush=True)
PY
cd "${FILTER_PROJECT_ROOT}"
case "${FILTER_PHASE}" in
  --check-cpu|--check-gpu-env) exit 0 ;;
  prepare)
    exec "${FILTER_PYTHON}" -u -m scripts.youtube_filter.prepare \
      --data-dir "${YOUTUBE_DATA_DIR}" --output-dir "${YOUTUBE_PILOT_DIR}" "$@" ;;
  dry-run)
    exec "${FILTER_PYTHON}" -u -m scripts.youtube_filter.score \
      --input-dir "${YOUTUBE_PILOT_DIR}" --output-dir "${YOUTUBE_SCORE_DIR}-dry-run" --dry-run "$@" ;;
  score)
    exec "${FILTER_PYTHON}" -u -m scripts.youtube_filter.score \
      --input-dir "${YOUTUBE_PILOT_DIR}" --output-dir "${YOUTUBE_SCORE_DIR}" "$@" ;;
  repair-evidence)
    exec "${FILTER_PYTHON}" -u -m scripts.youtube_filter.repair_evidence \
      --input-dir "${YOUTUBE_PILOT_DIR}" --source-dir "${YOUTUBE_SCORE_DIR}" \
      --output-dir "${YOUTUBE_REPAIR_DIR:-${YOUTUBE_SCORE_DIR}-evidence-repair}" "$@" ;;
esac

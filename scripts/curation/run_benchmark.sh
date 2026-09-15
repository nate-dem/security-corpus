#!/usr/bin/env bash
# Two model comparisons in one allocation; saved successful spans resume.
set -euo pipefail
CURATION_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CURATION_PROJECT"
export YOUTUBE_DATA_DIR="${YOUTUBE_DATA_DIR:-/scratch/m000091/${USER}/youtube-transcripts}"
export HF_HUB_CACHE="${YOUTUBE_MODEL_CACHE:-${YOUTUBE_DATA_DIR}/model-cache}"
export HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="${CURATION_PROJECT}/src:${CURATION_PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
CURATION_PYTHON="${YOUTUBE_FILTER_GPU_PYTHON:-${CURATION_PROJECT}/scripts/youtube_filter/.venv/bin/python}"
CURATION_INPUT="${CURATION_INPUT_DIR:-${CURATION_PROJECT}/reports/curation/development-v2}"
CURATION_OUTPUT="${CURATION_OUTPUT_DIR:-/scratch/m000091/${USER}/curation/development-v2}"
case "${1:-run}" in
  run|--check-env) ;;
  *) echo 'Usage: run_benchmark.sh [--check-env]' >&2; exit 2 ;;
esac
bash scripts/youtube_filter/run.sh --check-gpu-env
# Validate all source/reference hashes before requesting/using GPUs.
"$CURATION_PYTHON" - "$CURATION_INPUT" <<'PY'
import sys
from pathlib import Path
from scripts.curation.evaluate import evaluate
p=Path(sys.argv[1])
r=evaluate(p/'packet.json',p/'assistant_annotations.json')
print(f"Review inputs OK: {r['cases']} assistant-assessed cases; {r['reference_routes']}",flush=True)
PY
if [[ "${1:-run}" == --check-env ]]; then exit 0; fi
mkdir -p "$CURATION_OUTPUT"
for CURATION_SIZE in 8 32; do
  if [[ "$CURATION_SIZE" == 8 ]]; then
    CURATION_REVISION=b968826d9c46dd6066d109eabc6255188de91218
    CURATION_TP=1
  else
    CURATION_REVISION=9216db5781bf21249d130ec9da846c4624c16137
    CURATION_TP=2
  fi
  CURATION_STATUS=0
  "$CURATION_PYTHON" -u -m scripts.curation.score \
    --packet "$CURATION_INPUT/packet.json" --output-dir "$CURATION_OUTPUT/qwen${CURATION_SIZE}b" \
    --rubric-version "${CURATION_RUBRIC_VERSION:-v2}" \
    --model "Qwen/Qwen3-${CURATION_SIZE}B" --model-revision "$CURATION_REVISION" \
    --tensor-parallel-size "$CURATION_TP" || CURATION_STATUS=$?
  # Exit 2 denotes unresolved evidence. Still compare it; never relabel as drop.
  if [[ "$CURATION_STATUS" != 0 && "$CURATION_STATUS" != 2 ]]; then exit "$CURATION_STATUS"; fi
  "$CURATION_PYTHON" -u -m scripts.curation.evaluate \
    --packet "$CURATION_INPUT/packet.json" --reference "$CURATION_INPUT/assistant_annotations.json" \
    --score-dir "$CURATION_OUTPUT/qwen${CURATION_SIZE}b" \
    --output "$CURATION_OUTPUT/evaluation-qwen${CURATION_SIZE}b.json"
done
echo "COMPARISONS COMPLETE: $CURATION_OUTPUT. Inspect coverage and disagreements; this does not certify production accuracy."

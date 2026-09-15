#!/usr/bin/env bash
# Independent of the ongoing CPU preparation job; only small packets are read.
set -euo pipefail
CURATION_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CURATION_PROJECT"
export YOUTUBE_DATA_DIR="${YOUTUBE_DATA_DIR:-/scratch/m000091/${USER}/youtube-transcripts}"
export HF_HUB_CACHE="${YOUTUBE_MODEL_CACHE:-${YOUTUBE_DATA_DIR}/model-cache}"
export HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="${CURATION_PROJECT}/src:${CURATION_PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
CURATION_PYTHON="${YOUTUBE_FILTER_GPU_PYTHON:-${CURATION_PROJECT}/scripts/youtube_filter/.venv/bin/python}"
CURATION_REVIEWED="${CURATION_PROJECT}/reports/curation/development-v2"
CURATION_ADDITIONAL="${CURATION_PROJECT}/reports/curation/additional-v1"
CURATION_PRIOR="${CURATION_PRIOR_SCREEN:-/scratch/m000091/${USER}/curation/development-v3/qwen8b}"
CURATION_OUTPUT="${CURATION_REVIEW_OUTPUT:-/scratch/m000091/${USER}/curation/gpu-review-v1}"
case "${1:-run}" in run|--check-env) ;; *) echo 'Usage: run_review.sh [--check-env]' >&2; exit 2 ;; esac
bash scripts/youtube_filter/run.sh --check-gpu-env
"$CURATION_PYTHON" - "$CURATION_REVIEWED" "$CURATION_ADDITIONAL" "$CURATION_PRIOR" <<'PY'
import sys
from pathlib import Path
from scripts.curation.score import load_packet
from scripts.curation.evaluate import evaluate
old,new,prior=map(Path,sys.argv[1:])
r=evaluate(old/'packet.json',old/'assistant_annotations.json',prior)
p=load_packet(new/'packet.json')
print(f"Prior screening verified: {r['cases']} cases; additional packet: {len(p['cases'])} cases.",flush=True)
PY
if [[ "${1:-run}" == --check-env ]]; then exit 0; fi
for CURATION_PHASE in screen review; do
  "$CURATION_PYTHON" -u -m scripts.curation.review_run \
    --phase "$CURATION_PHASE" --reviewed-input "$CURATION_REVIEWED" \
    --additional-input "$CURATION_ADDITIONAL" --prior-screen "$CURATION_PRIOR" \
    --output-dir "$CURATION_OUTPUT"
done
echo "GPU REVIEW COMPLETE: $CURATION_OUTPUT. Check semantic results; no corpus selection was performed."

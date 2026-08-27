#!/bin/bash
#SBATCH --job-name=qwen-citations
#SBATCH --array=0-23%4
#SBATCH --account=marlowe-m000091-pm05
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/qwen_citations/%A_%a.out
#SBATCH --error=logs/qwen_citations/%A_%a.err

set -euo pipefail

# Citation-only arXiv title+abstract filtering.
# Input is the citation-only paper-level Parquet produced by ingest_arxiv.py
# after adding metadata/citations.

PROJECT_DIR="${PROJECT_DIR:-/scratch/m000091-pm05/natedem/security-corpus}"
INPUT="${INPUT:-data/arxiv/normalized/source_id=arxiv/raw.parquet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/filtering/v3/qwen_citation_abstract_shards}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
SHARD_ID="${SLURM_ARRAY_TASK_ID}"
NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT}"

cd "${PROJECT_DIR}"
source .venv/bin/activate
export PYTHONPATH="${PROJECT_DIR}/src"

mkdir -p "${OUTPUT_ROOT}" logs/qwen_citations

python scripts/classify/citation_paper_qwen_filtering.py \
  --input "${INPUT}" \
  --output "${OUTPUT_ROOT}/shard-${SHARD_ID}.jsonl" \
  --model "${MODEL}" \
  --tensor-parallel-size 1 \
  --dtype float16 \
  --batch-size 32 \
  --shard-id "${SHARD_ID}" \
  --num-shards "${NUM_SHARDS}"

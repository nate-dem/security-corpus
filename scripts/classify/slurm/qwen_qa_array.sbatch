#!/bin/bash
#SBATCH --job-name=qwen-qa-filter
#SBATCH --array=0-99
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/qwen_qa/%A_%a.out
#SBATCH --error=logs/qwen_qa/%A_%a.err

set -euo pipefail

# Submit manually after auditing dry-run prompts. Codex must not run sbatch.
# Before submission, create the Slurm log directory:
#   mkdir -p logs/qwen_qa

CORPUS_DIR="${CORPUS_DIR:-data/training-clean-v2/normalized}"
CANDIDATE_SIDECAR="${CANDIDATE_SIDECAR:-data/filtering/v3/qa_qwen_candidates.parquet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/filtering/v3/qwen_qa_shards}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-100}"
SHARD_ID="${SLURM_ARRAY_TASK_ID}"

mkdir -p "${OUTPUT_ROOT}/shard-${SHARD_ID}"

python scripts/classify/score_qwen_vllm.py \
  --input "${CORPUS_DIR}" \
  --task qa \
  --qa-sources \
  --candidate-sidecar "${CANDIDATE_SIDECAR}" \
  --candidate-keep-column qa_candidate_for_qwen \
  --output-dir "${OUTPUT_ROOT}/shard-${SHARD_ID}" \
  --model "${MODEL}" \
  --shard-id "${SHARD_ID}" \
  --num-shards "${NUM_SHARDS}" \
  --batch-size 32 \
  --max-tokens 96

#!/bin/bash
#SBATCH --job-name=qwen-arxiv-abstract
#SBATCH --array=0-99
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/qwen_arxiv_abstract/%A_%a.out
#SBATCH --error=logs/qwen_arxiv_abstract/%A_%a.err

set -euo pipefail

# Submit manually after auditing dry-run prompts. Codex must not run sbatch.
# Before submission, create the Slurm log directory:
#   mkdir -p logs/qwen_arxiv_abstract

CORPUS_DIR="${CORPUS_DIR:-data/training-clean-v2/normalized}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/filtering/v3/qwen_arxiv_abstract_shards}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-100}"
SHARD_ID="${SLURM_ARRAY_TASK_ID}"

mkdir -p "${OUTPUT_ROOT}/shard-${SHARD_ID}"

python scripts/classify/score_qwen_vllm.py \
  --input "${CORPUS_DIR}" \
  --task arxiv-abstract \
  --source-id arxiv \
  --output-dir "${OUTPUT_ROOT}/shard-${SHARD_ID}" \
  --model "${MODEL}" \
  --shard-id "${SHARD_ID}" \
  --num-shards "${NUM_SHARDS}" \
  --batch-size 32 \
  --max-tokens 96

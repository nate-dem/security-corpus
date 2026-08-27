#!/bin/bash
#SBATCH --job-name=qwen-qa-filter
#SBATCH --array=0-99
#SBATCH --account=marlowe-m000091-pm05
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/qwen_qa/%A_%a.out
#SBATCH --error=logs/qwen_qa/%A_%a.err

set -euo pipefail

# Submit manually after auditing dry-run prompts. Codex must not run sbatch.

PROJECT_DIR="${PROJECT_DIR:-/scratch/m000091-pm05/natedem/security-corpus}"
CORPUS_DIR="${CORPUS_DIR:-data/filtering/v4/qa_to_score}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/filtering/v4/qwen_qa_shards}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
MODEL_REVISION="${MODEL_REVISION:-b968826d9c46dd6066d109eabc6255188de91218}"
NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-100}"
SHARD_ID="${SLURM_ARRAY_TASK_ID}"

cd "${PROJECT_DIR}"
source .venv/bin/activate
export PYTHONPATH="${PROJECT_DIR}/src"

mkdir -p "${OUTPUT_ROOT}/shard-${SHARD_ID}" logs/qwen_qa

python scripts/classify/score_qwen_vllm.py \
  --input "${CORPUS_DIR}" \
  --task qa \
  --qa-sources \
  --output-dir "${OUTPUT_ROOT}/shard-${SHARD_ID}" \
  --model "${MODEL}" \
  --model-revision "${MODEL_REVISION}" \
  --shard-id "${SHARD_ID}" \
  --num-shards "${NUM_SHARDS}" \
  --batch-size 32 \
  --max-tokens 96

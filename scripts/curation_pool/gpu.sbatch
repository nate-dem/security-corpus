#!/usr/bin/env bash
#SBATCH --job-name=youtube-pool
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/curation/%x-%A_%a.log
set -euo pipefail
cd "${PROJECT_DIR:?Set PROJECT_DIR to the repository root}"
export PYTHONPATH="${PWD}/src:${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_CACHE="${CURATION_NEXT_MODEL_CACHE:-${CYBEROPEN_WORK_DIR:-${PWD}/data}/curation-model-cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="${PWD}/scripts/curation/.venv-next/bin:${PATH}"
exec scripts/curation/.venv-next/bin/python -u -m scripts.curation_pool.run work \
  --plan "${1:?}" --work-list "${2:?}" --job-index "${SLURM_ARRAY_TASK_ID:?}"

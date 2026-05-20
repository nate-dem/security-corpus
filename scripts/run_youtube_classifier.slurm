#!/bin/bash
# Slurm array job: classify all 439 YouTube-Commons shards on Marlowe HPC.
#
# Submit from the project root on Marlowe:
#   sbatch scripts/run_youtube_classifier.slurm
#
# Monitor progress:
#   squeue -u stanny04
#   squeue -j <jobid> --start
#
# After ALL tasks complete, merge caches locally or on a CPU login node:
#   python3 scripts/merge_classifier_caches.py --audit
#
# Then ingest:
#   python3 scripts/ingest_youtube_transcripts.py \
#       --cache-path data/youtube-transcripts/classifier_cache.jsonl
#
# Before sbatch: pre-download model weights and description dataset once on a
# login node (avoids 439 tasks racing to download simultaneously):
#
#   export HF_HOME="/scratch/m000091-pm05/stanny04/youtube-transcripts/.hf_cache"
#
#   # Model weights (~8 GB, no license gating):
#   python3 -c "
#   from huggingface_hub import snapshot_download
#   snapshot_download('Qwen/Qwen3-4B')
#   "
#
#   # Description enrichment dataset (~2.2 GB compressed):
#   python3 -c "
#   from huggingface_hub import snapshot_download
#   snapshot_download('Rijgersberg/YouTube-Commons-descriptions',
#                     repo_type='dataset',
#                     local_dir='/scratch/m000091-pm05/stanny04/youtube-transcripts/descriptions')
#   "
#
# Paths: this script assumes the repo is cloned to the location below.
# Adjust PROJECT_DIR and SCRATCH_DIR to match your environment.
# Shards that are already present in SCRATCH_DIR/raw/ are not re-downloaded.
#
# Time estimate: ~20–40 min per shard on a single A100 (varies with shard size
# and model load time).  2-hour wall time per task is conservative but safe.
# 439 tasks run in parallel, so total wall time ≈ time for the slowest shard.

#SBATCH --job-name=yt-classify
#SBATCH --array=0-438
#SBATCH --account=marlowe-m000091-pm05
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-gpu=16
#SBATCH --time=2:00:00
#SBATCH --output=logs/youtube-classify/shard_%a.out
#SBATCH --error=logs/youtube-classify/shard_%a.err

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_DIR="$HOME/security-corpus"
SCRATCH_DIR="/scratch/m000091-pm05/stanny04/youtube-transcripts"

RAW_DIR="$SCRATCH_DIR/raw"
CACHE_DIR="$SCRATCH_DIR/classifier_cache"
DESCRIPTIONS_DIR="$SCRATCH_DIR/descriptions"

# ── Environment ────────────────────────────────────────────────────────────
cd "$PROJECT_DIR"
source .venv/bin/activate

# HuggingFace cache goes to scratch so downloaded model weights survive
# across tasks (the first task to run downloads the model; subsequent tasks
# load it from the local cache directory).
export HF_HOME="$SCRATCH_DIR/.hf_cache"

# ── Create output directories ──────────────────────────────────────────────
mkdir -p "$RAW_DIR" "$CACHE_DIR" "$DESCRIPTIONS_DIR" logs/youtube-classify

# ── Run classifier for this array task's shard ────────────────────────────
# SLURM_ARRAY_TASK_ID is automatically set by Slurm to the shard index.
# --delete-after removes the raw shard parquet after classification to save
# scratch space (each shard is ~370 MB; 439 shards = ~163 GB total).
python3 scripts/classify_youtube_videos.py \
    --model             Qwen/Qwen3-4B \
    --raw-dir           "$RAW_DIR" \
    --cache-dir         "$CACHE_DIR" \
    --descriptions-path "$DESCRIPTIONS_DIR" \
    --batch-size        500 \
    --delete-after

# YouTube Transcript Classifier — Context File
# Last updated: 2026-05-20

---

## What Was Built

A vLLM-based metadata classifier that classifies YouTube videos as cybersecurity-relevant
using lightweight metadata (title, channel, word count, language, URL, optional description).
Replaces the old hard keyword allowlist gate in `src/ingest/connectors/youtube_transcripts.py`.

### Key files
- `scripts/classify_youtube_videos.py` — main classifier script (Slurm + local pilot modes)
- `scripts/run_youtube_classifier.slurm` — Slurm array job definition (439 tasks, one per shard)
- `src/ingest/connectors/youtube_classifier.py` — classifier module (schema, prompt, parsing, cache I/O)
- `src/ingest/connectors/youtube_transcripts.py` — ingest connector with three-gate filter
- `scripts/merge_classifier_caches.py` — merges 439 per-shard JSONL caches into one

---

## Model

**Qwen/Qwen3-4B** (switched FROM meta-llama/Llama-3.2-3B-Instruct)

Reason for switch: Llama-3.2-3B-Instruct is gated (requires Meta license approval on HuggingFace).
Qwen3-4B is non-gated, same parameter class, ~same throughput. Already handled by the
`enable_thinking=False` try/except in `_build_prompts()` and `<think>` stripping in `_parse_response()`.

Model weights location on Marlowe scratch: `/scratch/m000091-pm05/stanny04/youtube-transcripts/.hf_cache/`
(~8 GB, already downloaded)

---

## Scale (Real Numbers from Shard 0)

- **Candidates per shard**: ~33,554 (from shard 0 output; 439 shards × 33,554 ≈ **14.7M total candidates**)
- **Wall time per shard**: ~5.1 min on 1× A100 (not 20–40 min as originally estimated)
- **Total wall time (439 parallel, %16 concurrency)**: ~5–10 min wall time

### Initial classifier results (shard 0, OLD prompt — before fix)
- 91.5% retention rate — identified as broken
- 16,710 `not_relevant`-but-kept records (confidence < 0.50 forced keep) were clearly non-security
  (pronunciation tutorials, sermons, travel vlogs, nail care, cooking, math proofs)
- Root cause: model was using `relevance_level="low"` for clearly off-topic content;
  blanket `confidence < 0.50 → keep` rule prevented dropping them

### Prompt fix (2026-05-20, pushed to feature/regmix-midtraining)
- Tightened `"low"` definition: requires a PLAUSIBLE security angle, not just ambiguity
- Added explicit bad examples: "nail care, sermons, game walkthroughs, cooking, fitness, math proofs"
- New UNCERTAINTY RULE: requires plausible technical/security signal before applying; otherwise → `not_relevant`
- Drop threshold: `confidence >= 0.50` (was `0.70`) for `not_relevant`
- Constraints added directly to prompt end: `should_keep=false` when `not_relevant AND confidence >= 0.50`

**Validation needed**: Nate to rerun shard 0 with new prompt to confirm retention rate improvement.
Message to Nate:
```bash
git pull
rm /scratch/m000091-pm05/natedem/youtube-transcripts/classifier_cache/shard_0000.jsonl
sbatch --array=0-0 scripts/run_youtube_classifier.slurm
```

---

## Classifier Design

### Input (per video)
- Title
- Channel name
- Word count
- Language
- URL
- Description (optional, truncated to 300 chars — from Rijgersberg/YouTube-Commons-descriptions)

### System prompt
Lenient classifier with 17 positive cybersecurity categories. Key rules:
- False positives (keeping non-security) are acceptable
- False negatives (dropping valid security) are the primary risk to avoid
- `"low"` relevance requires a plausible security angle (NOT just uncertainty)
- Uncertainty rule: only applies when title/channel has plausible technical/security signal
- `should_keep=false` when `relevance_level="not_relevant"` AND `confidence >= 0.50`
- Parse failures default to `_LENIENT_FALLBACK` (keep=True, confidence=0.30)

### Output schema (ClassificationResult)
```python
is_cybersecurity_relevant: bool
confidence: float  # 0.0–1.0
relevance_level: Literal["high", "medium", "low", "not_relevant"]
reason: str        # ≤30 words
topic_tags: list[str]  # up to 5
should_keep: bool
```

### Cache format (per-shard JSONL)
One JSON object per line:
```json
{
  "video_id": "abc123",
  "channel_id": "UCxxx",
  "channel": "SecurityAcademy",
  "title": "SQL Injection Deep Dive",
  "word_count": 4200,
  "result": { "is_cybersecurity_relevant": true, "confidence": 0.91, ... },
  "classified_at": "2026-05-20T14:00:00Z",
  "model": "Qwen/Qwen3-4B"
}
```

---

## Description Enrichment

Dataset: `Rijgersberg/YouTube-Commons-descriptions` (HuggingFace, 2.23 GB compressed)
- Joinable on `video_id` with PleIAs/YouTube-Commons (same corpus we ingest)
- ~60–90% coverage of the 3M candidates
- Already downloaded to: `/scratch/m000091-pm05/stanny04/youtube-transcripts/descriptions/`
- Graceful degradation: videos without a description use the 5-field fallback

---

## Ingest Connector Changes (youtube_transcripts.py)

Three-gate filter in `iter_records()`:
1. **Language gate** — `_is_english()` with three-path BCP-47 logic
2. **Word count gate** — `>= MIN_WORD_COUNT (50)`
3. **Classifier gate** — `video_id in frozenset` from merged cache (O(1))

`_is_english()` three paths:
- Path 1: `transcription_language` matches `^en(-[A-Z]{2,3})?$`
- Path 2: `transcription_language` absent AND `source_language == "en"`
- Path 3: `language_id_method == "metadata"` AND `original_language == "en"` (garbage code fallback)

Two-pass column projection: metadata columns only in pass 1; `text` column fetched only for survivors.
Schema-resilient: `available_cols` intersects `_GATE_COLS` with actual parquet schema per file.

---

## Marlowe HPC Setup

### Cluster
- Login node: `login-01.marlowe.stanford.edu`
- SSH alias: `ssh marlowe` (config at `~/.ssh/config`, User=stanny04)
- Account (PI): Amin Saberi (`amin.saberi@gmail.com`)
- Project ID: `m000091`
- Slurm account: `marlowe-m000091-pm05` (medium/batch partition)
- Partition: `batch`

### Key paths on Marlowe
```
HOME:         /users/stanny04/
PROJECT_DIR:  /users/stanny04/security-corpus/
SCRATCH_DIR:  /scratch/m000091-pm05/stanny04/youtube-transcripts/
  ├── raw/                    # downloaded cctube_*.parquet shards (auto-deleted after classify)
  ├── classifier_cache/       # per-shard JSONL output (shard_0000.jsonl … shard_0438.jsonl)
  ├── descriptions/           # Rijgersberg/YouTube-Commons-descriptions parquet files
  └── .hf_cache/              # HuggingFace model cache (Qwen3-4B weights, ~8 GB)
```

### What's already done on Marlowe
- [x] Repo cloned to `~/security-corpus`, on branch `feature/regmix-midtraining`
- [x] venv created at `~/security-corpus/.venv`, vllm installed
- [x] HuggingFace token saved (`YOUTUBE_TOKEN`, stored at `~/.cache/huggingface/token`)
- [x] Qwen/Qwen3-4B weights downloaded to scratch `.hf_cache/` (~8 GB)
- [x] YouTube-Commons-descriptions dataset downloaded to scratch `descriptions/` (2.23 GB)
- [x] `logs/youtube-classify/` directory created
- [x] Group write permissions set on scratch dir (`chmod -R g+w /scratch/m000091-pm05/stanny04/youtube-transcripts/`)

### What's still blocked
- `stanny04` is NOT registered in Marlowe's Slurm accounting system
  - `sacctmgr show assoc user=stanny04` returns empty
  - `sbatch` fails: "Invalid account or account/partition combination specified"
  - Support email sent to `srcc-support@stanford.edu` (Sophia/Marcelo handling it)
  - Amin Saberi replied approving the request — waiting for Marlowe to process

---

## Slurm Job Details

```
Job name:    yt-classify
Array:       0–438 (439 tasks, one per cctube_*.parquet shard)
Account:     marlowe-m000091-pm05
Partition:   batch
GPUs:        1 per task (A100)
Memory:      100 GB per task
CPUs/GPU:    16
Wall time:   2 hours per task
```

All 439 tasks run in parallel (capped at %16 concurrent). Real wall time: ~5.1 min/shard → ~10 min total (slowest shard).

### Submit command (once account is fixed)
```bash
cd ~/security-corpus && source .venv/bin/activate
mkdir -p logs/youtube-classify
sbatch scripts/run_youtube_classifier.slurm
squeue -u stanny04
```

---

## Alternative: Nate Submitting

Nate (nate-dem on GitHub) has Marlowe access. He can submit on stanny04's behalf:
- Scratch paths still point to `/scratch/m000091-pm05/stanny04/youtube-transcripts/`
- Group write permissions set so Nate can write classifier cache there
- Nate needs: `git pull feature/regmix-midtraining`, `pip install vllm`, then `sbatch`

---

## Post-Job Steps

After all 439 tasks complete:
```bash
# Merge per-shard caches into one file
python3 scripts/merge_classifier_caches.py --audit

# Ingest (uses merged cache as gate 3)
python3 scripts/ingest_youtube_transcripts.py \
    --cache-path data/youtube-transcripts/classifier_cache.jsonl
```

---

## People & Contacts

| Person | Role | Username | Contact |
|---|---|---|---|
| Matthew Torre | User running the job | `stanny04` (Marlowe), `matthewtorre` (local) | matt.torre31@gmail.com |
| Abhinav Chinta | Lab collaborator, set up Marlowe | `achinta` | achinta@stanford.edu |
| Amin Saberi | PI, owns the allocation | — | amin.saberi@gmail.com |
| Shayan Talaei | Lab member | `stalaei` | stalaei@stanford.edu |
| Nate | Collaborator with Marlowe access | `nate-dem` (GitHub) | — |
| Sophia / Marcelo | Marlowe support staff | — | srcc-support@stanford.edu |

---

## Git

- Repo: https://github.com/nate-dem/security-corpus.git
- Branch: `feature/regmix-midtraining`
- Last relevant commit: `fadaed2` — "Switch YouTube classifier to Qwen/Qwen3-4B"

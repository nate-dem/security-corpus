# Start the YouTube content-filtering pilot

**Next implementation (2026-09-14):** full English preparation and a separate
source-specific development classifier are now available. See the
[quality-first curation protocol](../../docs/curation_protocol.md). The old
pilot/repair commands below are not the next jobs to submit.

**Current Marlowe state (2026-09-14):** repair job 485708 generated all 76
requests, repairing 70. Combined with 561 original valid results, 631/637 spans
have valid evidence. Six spans across four transcripts remain scope/evidence
disagreements. See the [repair review](../../docs/youtube_repair_review.md).
Do not resubmit the unchanged repair job; the next work is classification review.
The [32B review](../../docs/youtube_32b_review.md) documents the earlier comparison.
The initial setup below documents v1 for reproducibility.

This can run while the separate web-corpus job runs. The raw YouTube download
and metadata profile are already complete. This folder supplies the missing
transcript-content classifier and a bounded pilot to measure its behavior.

| Script | Purpose | Compute |
|---|---|---|
| `prepare.py` / `prepare.sbatch` | Draw 500 random raw English rows plus up to two English variants for each of 13 previously inspected cases; retrieve full text, compute hashes/tokens, and preserve duplicate lineage. | 2 CPUs, 8 GB |
| `rubric.py` | Define content labels, split full transcripts into contiguous spans, and validate exact evidence quotes. | Used by scorer |
| `score.py` / `pilot.sbatch` | Run pinned Qwen3-8B over every pilot span and save resumable labels, evidence offsets, raw responses, and coverage/throughput reports. | 1 GPU, 8 CPUs, 64 GB |
| `repair_evidence.py` / `repair_v3.sbatch` | Preserve original v3 labels and request missing citations in a separate sidecar. Completed as job 485708; see the review above. | 2 GPUs, 8 CPUs, 128 GB |
| `audit_repair.py` | Reproduce original/repair requests, validate saved results, and extract unresolved spans for local review. | CPU and cached tokenizer; no inference |
| `run.sh` | Check the appropriate environment and call a stage with the Marlowe paths. | No separate job |
| `requirements-*.txt` | Pin the CPU/GPU dependencies. | Installation only |
| `pilot_cases.json` | Previously inspected video IDs for diagnostic enrichment; never a production allowlist. | No separate job |

For comparison, the **web-corpus** workflow uses `download.py` for verified raw
files, `profile.py` for exact token/hash inventories and diagnostic samples,
and `compare.py`/`compare_baseline.sh` for exact overlap. Its `job.sbatch` and
`run.sh` run those stages. It has not yet applied LLM filtering.

## What the YouTube pilot judges

The researcher approved **English, including translations**, and **security
plus directly supporting technical material**. The classifier judges actual
transcript content for security relevance, technical substance, text usability,
language, and mixed content. It receives no titles, channels, or popularity
metadata. The earlier metadata classifier and 50-word minimum are not used.

`prepare.py` selects the exact `en` metadata label for the initial pilot and
retains original-language metadata. It imposes no content-length threshold.
Empty/missing text is accounted for separately. Identical nonempty text is
scored once within the pilot, with all selected source locations retained.
Different texts with the same video ID remain separate candidates.

500 is a configurable initial pilot workload (`--sample-size`), not a quality
threshold or a promise of sufficient validation. The random arm and enriched
diagnostic cases remain distinguishable. Raw-row sampling is affected by
duplicate/translation clusters. Do not estimate corpus yield from pooled,
enriched, or exact-deduplicated pilot rows without their sampling lineage.

The scorer pins `Qwen/Qwen3-8B` at
`b968826d9c46dd6066d109eabc6255188de91218`, with the same tokenizer revision.
It uses vLLM 0.13.0, Transformers 4.57.3, thinking disabled, and schema-constrained
JSON. Default settings are 4,096 model tokens per focal span, neighboring text
of up to 1,024 characters on each side, an 8,192-token total context, and a
1,024-token response allowance. These are adjustable inference budgets, not
document eligibility limits. Every rendered prompt is measured and oversized
spans are subdivided; no transcript tail is silently truncated.

Focal spans partition the entire original string without gaps or overlap.
Neighbor context is provided for interpretation but not credited again in
coverage counts. Corpus `content_length` is computed once for each complete
text with cl100k_base; model input/output tokens are separate measurements.
Character counts in the pilot reports measure exact text coverage, not corpus
token length.

Evidence quotes must occur in the focal text, with a zero-based occurrence
index when repeated. The runner resolves their character offsets. Parsing,
truncation, missing-evidence, and coverage failures remain unresolved. Every
raw attempt is preserved. A successful parser/evidence check does not establish
that the model's judgment is correct.

**This is the first LLM filtering pilot, not the final selection.** It writes
labels with `selection_decision: null`. We will review the labels, failure cases,
and throughput before choosing production keep rules and allocating a full GPU
run. No canonical transcript schema or production thresholds are introduced.
The existing cleaned baseline remains the basis for the 3B retained-token goal.

## 1. Copy only this new folder from the Mac

The web job is already running against the main checkout. This transfer adds
the YouTube pilot without changing its implementation or dependency environment:

```bash
rsync -av --exclude='.venv/' --exclude='__pycache__/' \
  /Users/natedemchak/Desktop/security-corpus/scripts/youtube_filter/ \
  natedem@login.marlowe.stanford.edu:/scratch/m000091/natedem/security-corpus/scripts/youtube_filter/
```

The new folder uses the existing `src/ingest/utils.py` and `scripts/youtube/`
helpers already transferred with the web setup. Bulk data and model weights
stay on Marlowe; no full datasets need downloading to the Mac.

## 2. Start CPU pilot preparation now

On Marlowe, from a free shell prompt in the existing tmux session or another
SSH terminal:

```bash
cd /scratch/m000091/natedem/security-corpus
export YOUTUBE_DATA_DIR=/scratch/m000091/natedem/youtube-transcripts

bash scripts/youtube_filter/run.sh --check-cpu && \
mkdir -p logs/youtube-filter && \
sbatch scripts/youtube_filter/prepare.sbatch
```

This reuses the **web CPU virtual environment**, which already contains all
three required packages. It reads the existing YouTube metadata index, checks
its provenance, and retrieves only selected transcript texts. It does not redo
the full download/profile. Completed per-shard extractions are reused after
interruption. Output: `$YOUTUBE_DATA_DIR/pilot-v1/`.

Save the job ID from `sbatch`. The two-hour allocation is a time limit, not a
runtime estimate. Its CPU job is independent of the running web job; actual
start time depends on Slurm availability.

## 3. Prepare the GPU environment while the CPU job runs

On Marlowe, install a separate environment; do not install GPU dependencies
into the active web environment:

```bash
cd /scratch/m000091/natedem/security-corpus
python3 -m venv scripts/youtube_filter/.venv
scripts/youtube_filter/.venv/bin/python -m pip install \
  -r scripts/youtube_filter/requirements-gpu.txt
bash scripts/youtube_filter/run.sh --check-gpu-env
```

Python 3.10–3.13 is supported by the pinned stack. Installation requires several GB
on Marlowe for PyTorch/vLLM dependencies. This environment check imports the
libraries but does not load model weights or require a GPU allocation. Actual
CUDA execution is checked in the pilot job; local Mac tests cannot establish
Marlowe driver compatibility.

Qwen weights are first downloaded into `$YOUTUBE_DATA_DIR/model-cache` on
Marlowe when the GPU job starts (roughly 16–20 GB for the 8B weights plus cache
metadata). The token-count vocabulary is already packaged offline. Hugging Face
authentication paths are preserved; Qwen3-8B itself is public.

## 4. Run the GPU pilot after preparation completes

When the CPU job prints `PILOT READY`, and the GPU environment check passes:

```bash
sbatch scripts/youtube_filter/pilot.sbatch
```

The template requests **one GPU, eight CPUs, 64 GB RAM, two hours** on the
working `marlowe-m000091` / `preempt` account/partition. It does not use the
disabled `pm05` allocation. The web job may continue alongside it.

You may instead submit with a dependency before the CPU preparation finishes;
replace `PREP_JOB_ID` with its actual job number:

```bash
sbatch --dependency=afterok:PREP_JOB_ID scripts/youtube_filter/pilot.sbatch
```

Use one GPU submission method. If preparation is preempted and resubmitted as
a new job, the old dependency will not follow the replacement job; cancel the
still-pending dependent job and submit again with the successful preparation ID.
For the simplest workflow, wait for `PILOT READY` and use the first command.

Monitor with:

```bash
squeue -u natedem
# Replace JOBID with the GPU job number:
tail -f logs/youtube-filter/youtube-qwen-pilot-JOBID.log
```

If preempted, resubmit the same stage. Valid span decisions are revalidated and
reused. Failed spans remain eligible for retry; previous attempts are retained.
Configuration changes require a new `YOUTUBE_SCORE_DIR` so different models,
prompts, dependencies, or context settings cannot silently share decisions.
Changing only preparation sample size similarly requires a new
`YOUTUBE_PILOT_DIR`.

The pilot uses eager execution to reduce compilation startup time. Its measured
throughput describes those settings; do not extrapolate a production GPU array
without reviewing the workload and throughput report. Persistent invalid labels
need prompt/model diagnosis, not repeated identical deterministic retries.

## 5. Copy back only pilot/review outputs

On the Mac after the pilot completes or reports unresolved spans:

```bash
mkdir -p /Users/natedemchak/Desktop/security-corpus/reports/youtube/pilot-v1
mkdir -p /Users/natedemchak/Desktop/security-corpus/reports/youtube/pilot-qwen8b-v1

rsync -av --include='/summary.json' --include='/candidates.jsonl' \
  --include='/lineage.jsonl' --include='/run-config.json' --exclude='*' \
  natedem@login.marlowe.stanford.edu:/scratch/m000091/natedem/youtube-transcripts/pilot-v1/ \
  /Users/natedemchak/Desktop/security-corpus/reports/youtube/pilot-v1/

rsync -av --include='/summary.json' --include='/document_report.jsonl' \
  --include='/run-config.json' --include='/decisions/***' --exclude='*' \
  natedem@login.marlowe.stanford.edu:/scratch/m000091/natedem/youtube-transcripts/pilot-qwen8b-v1/ \
  /Users/natedemchak/Desktop/security-corpus/reports/youtube/pilot-qwen8b-v1/
```

These are bounded pilot records and decisions, not the 163 GB source dataset or
model cache. Reports distinguish complete evidence-validated coverage from
uncertain labels and from final selection, which is still undecided. Inspect
both favorable and unfavorable predictions and mixed-content spans. Case IDs,
titles, channels, and source metadata are present for human review but were
excluded from model prompts.

## Local validation and boundaries

CPU tests cover sampling translations and very short rows, provenance checks,
deduplication with lineage, full Unicode span coverage, context-budget failure,
literal role-marker escaping, exact/repeated evidence quotes, malformed/truncated
responses, retry preservation, and incomplete-document accounting. Local validation
passed 327 unit tests (10 real-data tests excluded), Ruff, and shell syntax checks.
An offline dry run with the actual pinned Qwen tokenizer planned 78 spans covering
all 58 available English profile samples, without model weights. Those diagnostic
samples are not a sufficient labeled evaluation set.

A `score.py --dry-run` loads the pinned tokenizer only and writes the complete
rendered requests. It does not import vLLM or download model weights. GPU inference
and actual classification quality still require the Marlowe pilot.

References: [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B),
[vLLM structured outputs](https://docs.vllm.ai/en/v0.13.0/features/structured_outputs/),
[vLLM installation](https://docs.vllm.ai/en/v0.13.0/getting_started/installation/gpu/),
[Marlowe Slurm](https://marlowe-research.stanford.edu/documentation/slurm/).

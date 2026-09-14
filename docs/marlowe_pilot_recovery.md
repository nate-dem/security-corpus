# Marlowe recovery after the first web and YouTube pilot runs

Updated 2026-09-14. The researcher reported recovery job **482460** completed
with exit 0 in 6m52s, and YouTube v2 job **482462** completed with exit 0 in 8m55s.
The reports have now transferred and passed integrity checks. Semantic review
found continuing 8B relevance errors; see [results and the 32B comparison
job](expansion_results_review.md). Do not resubmit these completed jobs. The
recovery commands remain documented here for reproducibility.

## What happened

| Job | Result | Cause and recovery |
|---|---|---|
| 479524: web corpus | Failed after RedSage completed | Primus stores text in `content`; the original reader expected `text`. Profile Primus into `profile-v2`, then compare it with RedSage's existing `profile-v1`. |
| 479747: YouTube preparation | Completed | 511 selected rows, 4 blank/missing rows, and 20 extra exact copies yielded 487 distinct texts and 1,140,848 candidate cl100k_base tokens. Reuse these inputs. |
| 479754: Qwen3-8B pilot | Inference finished; exit 2 for incomplete validation | All 637 spans ended normally. 387 passed validation; 250 contained at least one quote/occurrence absent from focal text. Run the bounded pilot with segment-reference evidence and preserve the original decisions. |

The web log reports **11,445,384,960 raw cl100k_base tokens** for RedSage,
also exact-unique within that source. This is candidate volume before semantic
filtering, near-deduplication, or baseline/cross-source overlap. It does not
increase the retained corpus count yet. Primus's actual fields were confirmed
from the pinned shard on Marlowe: `source`, `url`, `content`, `time`.
The source mapping preserves complete text and diagnostic raw metadata; absent
page licenses, upstream scores, and upstream token counts remain absent.

The YouTube audit verified candidate hashes/token lengths, request hashes using
the pinned tokenizer, decision-to-request correspondence, and complete focal
coverage. All 250 failed spans were invalid evidence, not output truncation.
Across them, 437 quotes failed: 116 wrong occurrence numbers, 77 whitespace-only
differences, 23 case-only differences, 6 combined case/whitespace differences,
22 quotes outside the focal span, and 193 other mismatches, including paraphrases
and invented quotations. A span can have multiple failures. These are validation
failures, not quality rejection labels or evidence of lost input data.

Local diagnostic outputs are `reports/youtube/pilot-review-v1/audit.json` and
`positive_span_review.jsonl`. Keep the original requests, decisions, input texts,
and run configuration as the evidence underlying the review.

## What changes in the second YouTube pilot

The v2 prompt presents the complete focal text in numbered segments of at most
400 characters. This is an evidence-display budget, not a filtering threshold.
Generation is constrained to IDs actually present in that request. Python resolves
the cited IDs to exact source text and offsets; the model no longer generates
quotations or counts repeated occurrences. All text, including whitespace, remains
covered. Neighbor context is not eligible evidence. The runner still checks labels,
required positive evidence, truncation, and complete coverage, and reports failure
reasons explicitly. Resolving a real passage does not prove it supports the label.

The first pilot also showed relevance drift toward general crime/physical safety
and a tendency to downgrade text usability solely because content was off-topic.
The v2 wording clarifies cybersecurity scope and independent dimension assessment.
It preserves the approved security plus directly supporting technical scope.
No production keep rules, quality thresholds, or canonical schema are changed.

Run the **same 487-document pilot** again so successful v1 outputs can also be
compared with v2. Do not pool changed-prompt decisions as one uniform evaluation.
This is a bounded calibration rerun, not a run over all 3.26 million English rows.
The first pilot used about 21 minutes, including about 18 minutes of generation;
that measurement is not a guarantee of v2 runtime or queue delay.

## Copy the fixes from the Mac

The three previous jobs are no longer running. Copy these two script directories
without replacing the installed virtual environments:

```bash
rsync -av --exclude='.venv/' --exclude='__pycache__/' \
  /Users/natedemchak/Desktop/security-corpus/scripts/web_corpora \
  /Users/natedemchak/Desktop/security-corpus/scripts/youtube_filter \
  natedem@login.marlowe.stanford.edu:/scratch/m000091/natedem/security-corpus/scripts/
```

No package reinstall or dataset/model download is needed for these fixes.

## Submit both jobs on Marlowe

```bash
cd /scratch/m000091/natedem/security-corpus
export WEB_DATA_DIR=/scratch/m000091/natedem/web-corpora
export YOUTUBE_DATA_DIR=/scratch/m000091/natedem/youtube-transcripts
mkdir -p logs/web-corpora logs/youtube-filter

bash scripts/web_corpora/run.sh --check-env && \
sbatch scripts/web_corpora/resume_primus.sbatch

bash scripts/youtube_filter/run.sh --check-gpu-env && \
sbatch scripts/youtube_filter/pilot_v2.sbatch
```

The jobs are independent. The web job requests 4 CPUs/16 GB for up to four hours;
the YouTube job requests 1 GPU/8 CPUs/64 GB for up to two hours. Both use the
working `marlowe-m000091` account and `preempt` partition. The completed YouTube
preparation needs no dependency job or resubmission.

The login-node GPU import check can still print the no-active-driver Triton
message. Actual GPU availability is checked by `nvidia-smi` in the allocated job.

The web job writes a web-only overlap report before attempting baseline overlap.
If baseline files are missing, the log names the missing path; the completed
profiles and web-only comparison remain usable. Copy the retained baseline with
`scripts/marlowe/sync_all.sh`, then submit only the baseline comparison:

```bash
sbatch scripts/web_corpora/job.sbatch compare-baseline \
  --source-profile primus-fineweb=profile-v2 \
  --output /scratch/m000091/natedem/web-corpora/comparison-baseline-v2.json
```

## Outputs and restart behavior

- Web logs: `logs/web-corpora/web-primus-recovery-JOBID.log`.
- Primus profile: `$WEB_DATA_DIR/primus-fineweb/profile-v2/`.
- Web comparisons: `$WEB_DATA_DIR/comparison-v2.json` and `comparison-baseline-v2.json`.
- YouTube log: `logs/youtube-filter/youtube-qwen-v2-JOBID.log`.
- YouTube outputs: `$YOUTUBE_DATA_DIR/pilot-qwen8b-v2/`.

After preemption, resubmit the same new job script. Completed Primus shards and
valid v2 span decisions are revalidated and reused. Do not use the old full web
job to recover: its default `profile-v1` conflicts with the changed reader code.
Likewise, do not point the changed scorer at `pilot-qwen8b-v1`; the version checks
intentionally prevent mixing old and new configurations. The v1 rubric remains
unchanged for auditing its saved responses.

Use `squeue -u natedem` while jobs are pending/running and `sacct` after they leave
the queue. A successful YouTube run still requires semantic review; `complete`
means valid output/evidence coverage, not approved final corpus selection.

Copy back the new YouTube outputs from the Mac when the GPU job ends:

```bash
rsync -av \
  natedem@login.marlowe.stanford.edu:/scratch/m000091/natedem/youtube-transcripts/pilot-qwen8b-v2/ \
  /Users/natedemchak/Desktop/security-corpus/reports/youtube/pilot-qwen8b-v2/
```

Copy the web reports and diagnostic samples from the Mac after that job ends:

```bash
mkdir -p /Users/natedemchak/Desktop/security-corpus/reports/web-corpora
rsync -av \
  --include='/redsage-cfw/manifest.json' \
  --include='/redsage-cfw/download-summary.json' \
  --include='/redsage-cfw/' --include='/redsage-cfw/profile-v1/' \
  --include='/redsage-cfw/profile-v1/summary.json' \
  --include='/redsage-cfw/profile-v1/run-config.json' \
  --include='/redsage-cfw/profile-v1/inspection_sample.jsonl' \
  --include='/primus-fineweb/manifest.json' \
  --include='/primus-fineweb/download-summary.json' \
  --include='/primus-fineweb/' --include='/primus-fineweb/profile-v2/' \
  --include='/primus-fineweb/profile-v2/summary.json' \
  --include='/primus-fineweb/profile-v2/run-config.json' \
  --include='/primus-fineweb/profile-v2/inspection_sample.jsonl' \
  --include='/comparison-v2.json' --include='/comparison-baseline-v2.json' \
  --exclude='*' \
  natedem@login.marlowe.stanford.edu:/scratch/m000091/natedem/web-corpora/ \
  /Users/natedemchak/Desktop/security-corpus/reports/web-corpora/
```

These transfers exclude raw shards, metadata sidecars, temporary aggregation
files, model weights, and virtual environments.

## Local verification

338 unit tests passed; 10 tests requiring full local corpus data were excluded.
Ruff passed. Tests cover the actual Primus field structure, missing/blank/invalid
text accounting, reuse of RedSage with a different Primus profile version, exact
segment resolution including repeated text and Unicode, invalid evidence IDs,
per-request generation constraints, and cached-result revalidation.

An offline dry run using the actual pinned Qwen tokenizer planned **637 spans
covering all 487 pilot transcripts** with the v2 prompt. GPU generation with the
new prompt was subsequently completed on Marlowe; its semantic limitations are
documented in the results review linked above. All 13,958
nonblank evidence segments were also resolved and checked against the exact
original source text/offsets locally. The largest rendered prompt used 5,969
model tokens, leaving room for the 1,024-token response within the 8,192 limit.

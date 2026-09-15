# Quality pass and retry — completed

Job **487038 completed successfully**, and the transferred archive, raw response
and original-run bindings have been verified. With the linked retry, all
**1,172 spans across 1,015 documents** now parse. The retried transcript remains
`review_required` because of corrupted technical terms. The other completed
assessments are unchanged. See the [verified review](../../docs/quality_pass_review.md).

**Do not submit these jobs again.** The commands below document completed work.
No additional environment setup or formatting retry is pending. Remaining work
is production scoring/selection, quality auditing, deduplication and release
assembly; see [the project status](../../docs/finish_plan.md). These diagnostic
results are not final corpus additions.

## Historical commands: one-response retry

Job **487003 finished all five packets**. One of 1,172 responses hit its output
limit in a carriage-return loop; the other 1,171 parsed. The failed status is
intentional incomplete-coverage reporting. See the
[verified results and remaining quality findings](../../docs/quality_pass_review.md).

Copy code on the Mac:

```bash
cd /Users/natedemchak/Desktop/security-corpus
bash scripts/curation/sync_first_batch.sh
```

Then submit on Marlowe:

```bash
cd /scratch/m000091/natedem/security-corpus
bash scripts/curation/submit_quality_pass.sh --retry
```

The helper re-parses the original quality-pass artifacts and verifies the
existing runtime. The plan should show one case and one expected span under
`part-009211`. The job requests **one H100, 8 CPUs, 96 GB RAM, a 30-minute cap**.
It uses the cached model and native sampler with compact JSON decoding. No new
installation or download is needed. The source text and prompt remain intact;
outputs go to a separate `quality-retry-v1/qwen38-27b-fp8/` directory. Original
quality-pass outputs remain unchanged. This completes a diagnostic assessment,
not production filtering or final selection.

After the retry ends, run on the Mac:

```bash
mkdir -p /Users/natedemchak/Desktop/security-corpus/reports/curation/quality-retry-v1
rsync -av \
  --include='/*/' --include='/*/summary.json' \
  --include='/*/bundle-summary.json' --include='/*/review-bundle.tar.gz' \
  --exclude='*' \
  natedem@login-01.marlowe.stanford.edu:/scratch/m000091/natedem/curation/quality-retry-v1/ \
  /Users/natedemchak/Desktop/security-corpus/reports/curation/quality-retry-v1/
```

If it fails before producing a bundle, copy
`logs/curation/curation-quality-retry-JOBID.log` too. On preemption the same
retry command can resume checkpoints with unchanged code and settings.
Do not resubmit the completed full quality pass below; its old configuration
is preserved as provenance.

## Historical run: source-only quality recheck

First-batch job **486937 completed**. The [review](../../docs/first_batch_review.md)
prefers Qwen3.8-27B-FP8 for the next assessment, while identifying real errors
among its eligible candidates. No more environment setup is needed.

## Copy code on the Mac

```bash
cd /Users/natedemchak/Desktop/security-corpus
bash scripts/curation/sync_first_batch.sh
```

This transfers code and the small existing control files. The first-batch packets
and raw outputs already exist on Marlowe; they are read there directly.

## Submit one GPU job on Marlowe

```bash
cd /scratch/m000091/natedem/security-corpus
bash scripts/curation/submit_quality_pass.sh
```

The submission helper re-parses the prior 27B results, verifies bindings and the
installed runtime, then submits one job: **1 H100, 8 CPUs, 96 GB RAM, 4-hour cap**.
It uses the successful native sampler/compiler preflight and cached model. It
does not need a new CPU preparation job, downloads, an active tmux session or
additional model installations. The full-pipeline runtime is not yet measured.

The model stays loaded for all five packets:

| Packet | Documents |
|---|---:|
| Original development controls | 89 |
| Previously accepted-case controls | 48 |
| Primus eligible/review + 12 sampled rejections | 486 |
| RedSage eligible/review + 12 sampled rejections | 314 |
| YouTube eligible/review + 12 sampled rejections | 78 |
| Total | 1,015 |

The `critic-v2` prompt rechecks the complete supplied source, using literal
commands/definitions and the existing quality dimensions. It receives no prior
model labels. Four synthetic demonstrations clarify short useful explanations
versus broken source and mismatched impacts. Output allowance increases to
1,536 tokens; model revision, precision, temperature zero and native sampler
remain the same. Raw v4 outputs and historical rubric modules stay intact.

The rejection sample size is a diagnostic workload choice, not a content filter.
All old uncertain/unparsed documents are included. No document is selected for
release and no unsampled document is deleted. Agreement between two calls to the
same model has correlated errors and cannot establish independent accuracy.

Outputs:
`/scratch/m000091/natedem/curation/quality-pass-v1/qwen38-27b-fp8/`.
The job writes re-parsed evaluations, first/second-pass comparisons and a small
review bundle. It exits 2 if any new span remains unresolved, preserving the
bundle for inspection; a failed span never becomes a keep/drop decision. On
preemption, resubmit the same command with unchanged code and output directory
to resume successful span checkpoints.

## Return reports to the Mac

After the job ends:

```bash
mkdir -p /Users/natedemchak/Desktop/security-corpus/reports/curation/quality-pass-v1
rsync -av \
  --include='/*/' --include='/*/summary.json' \
  --include='/*/bundle-summary.json' --include='/*/review-bundle.tar.gz' \
  --exclude='*' \
  natedem@login-01.marlowe.stanford.edu:/scratch/m000091/natedem/curation/quality-pass-v1/ \
  /Users/natedemchak/Desktop/security-corpus/reports/curation/quality-pass-v1/
```

If the job fails before creating a bundle, copy its
`logs/curation/curation-quality-pass-JOBID.log`. The review determines whether
the second pass catches actual damage while retaining useful text, then informs
production scoring and the final selection workflow. No user labeling task is
pending, but assistant development references are not independent human evidence.

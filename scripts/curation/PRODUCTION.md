# Run the first production batch

Model comparison and formatting recovery are complete. This workflow scores
prepared partitions, exports candidate Parquet files, and audits them against
raw responses and source records. No installation or download is needed.

## Copy code from the Mac

```bash
cd /Users/natedemchak/Desktop/security-corpus
REMOTE_HOST=natedem@login-01.marlowe.stanford.edu \
  bash scripts/marlowe/sync_code.sh
```

This copies code and tokenizer assets, not bulk data or virtual environments.

## Submit on Marlowe

```bash
cd /scratch/m000091/natedem/security-corpus
bash scripts/curation/submit_production.sh
```

The helper validates the existing runtime, freezes the input/code/model plan,
and submits **eight GPU array tasks with at most two running at once**. Each
requests **one H100, eight CPUs, 96 GB RAM and a four-hour cap**, using account
`marlowe-m000091` and partition `preempt`. A dependent **CPU audit** requests
four CPUs, 16 GB RAM and a two-hour cap. It starts after the array ends, including
failures, so partial progress is reported. The jobs do not need an active tmux
session or SSH connection after submission.

Four partitions from distinct original shards are selected per source, excluding
the source shards used in the earlier three-part comparison. Slots alternate
between sources. The transferred prepared manifest gives:

| Source | Prepared partition indices | Documents | Input reference tokens |
|---|---|---:|---:|
| Primus | 537, 1017, 886, 721 | 7,795 | 5,608,861 |
| YouTube | 8907, 10609, 9900, 9771 | 5,431 | 12,890,190 |
| Total | Eight partitions | 13,226 | 18,499,051 |

These are input tokens, not retained additions. The partition count allocates
work; it is not a relevance threshold or a representative yield estimate.
RedSage remains prepared for subsequent work.

## Scoring and candidate selection

Each task verifies its input checksum and every document's text hash and token
count. It runs `v4` and `critic-v2` over **every complete document**, keeping one
model resident for both passes. The critic sees source text without the first
verdict. Both use `Qwen/Qwen3.8-27B-FP8`, revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, compact JSON, native sampling,
temperature zero, 8,192-token context and a 1,536-token output allowance.

The existing policy is unchanged: a whole document enters the candidate export
only when every span is eligible under both passes. Joint exclusions remain
exclusions. Disagreements, mixed span routes and parse failures remain review
items. No source passage is reconstructed, shortened or deleted. Two calls to
the same model can share mistakes; their agreement is not independent accuracy.

Each pass automatically retries unresolved responses once, reusing successful
checkpoints. Persistent failures stay held and cause exit 2; completed work is
still exported and available for review. Resubmission retries unfinished work.

## Outputs and automatic audit

Default root: `/scratch/m000091/natedem/curation/production-v1/`.
Each `parts/0000/` through `parts/0007/` directory contains:

- `documents.parquet`: every source row, with the prepared schema unchanged.
- `candidates.parquet`: exact source rows jointly eligible under both assessments.
- `decisions.parquet`: document routes, coverage and configuration hashes, with
  final selection unset.
- `spans.parquet`: queryable quality dimensions and raw response attempts with
  model/revision, prompt version, offsets and provenance hashes.
- `summary.json`: output checksums, counts and generation accounting.
- `work/`: original resumable scoring checkpoints, kept on Marlowe for this batch.

The plan binds `inputs-v1/lineage.parquet`, which retains every source alias;
the CPU audit checks its checksum. Candidate files retain source locators and
dataset revisions. They do not assert new license permissions or constitute the
canonical public release schema. Attribution and release assembly remain pending.

The CPU job re-parses raw decisions, verifies full span coverage, checks candidate
membership against the input and both passes, and recomputes reference token
counts. It reports exact duplicate candidate copies within this batch without
choosing source precedence or claiming final additions.

It also writes a review packet of up to **20 complete documents per source/route**
(eligible, excluded, review), taking small strata in full. The sample size is an
audit workload setting. The packet omits model verdicts; `audit-key.jsonl` stores
them separately. Assistant review will be disclosed as such. The sample does
not establish population precision/recall, and `complete` does not mean quality
review or release approval is complete.

## Monitor and return reports

On Marlowe:

```bash
squeue -u natedem
cat /scratch/m000091/natedem/curation/production-v1/submitted-jobs.txt
```

The helper prints both job IDs. Logs are
`logs/curation/corpus-production-ARRAYID_SLOT.log` and
`logs/curation/corpus-production-audit-JOBID.log`.

After the CPU audit ends, run on the Mac:

```bash
mkdir -p /Users/natedemchak/Desktop/security-corpus/reports/curation/production-v1
rsync -av \
  natedem@login-01.marlowe.stanford.edu:/scratch/m000091/natedem/curation/production-v1/review/ \
  /Users/natedemchak/Desktop/security-corpus/reports/curation/production-v1/
```

This copies only summaries and the bounded review packet/bundle. If a worker or
audit fails, also copy its log. After fixing the cause, the **same submission
command retries only incomplete slots** and reruns the CPU audit. It verifies
completed output checksums and refuses a duplicate while recorded jobs are active.
Keep code, plan and environment unchanged during a run; changes require a new
output directory.

After this batch, review accepted/rejected examples and measured throughput before
choosing the next allocation. Full-corpus exact/near deduplication, final retained
counts, attribution, release assembly and Hugging Face publication remain pending.
The 1.468B-token baseline is unchanged by these provisional candidate exports.

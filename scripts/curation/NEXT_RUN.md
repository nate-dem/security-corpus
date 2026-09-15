# Run after the v2 comparison

**Update:** v3 reports have been reviewed and CPU preparation is still running.
Use [GPU_REVIEW.md](GPU_REVIEW.md) for the independent GPU follow-up. Leave the
CPU job running; the submission commands below describe the prior step.

The transferred v2 comparison exposed false quality flags and scope errors.
See [the findings](../../docs/curation_benchmark_review.md). Do not repeat the old
v2 job. The following jobs are independent and can run concurrently.

## Copy from the Mac

```bash
cd /Users/natedemchak/Desktop/security-corpus
bash scripts/curation/sync_to_marlowe.sh
```

This copies code and the unchanged 89 reviewed cases. No bulk data or model
weights are downloaded to the Mac.

## Submit on Marlowe

```bash
cd /scratch/m000091/natedem/security-corpus
bash scripts/youtube_filter/run.sh --check-cpu &&
bash scripts/curation/run_benchmark.sh --check-env &&
mkdir -p logs/curation &&
sbatch scripts/curation/prepare.sbatch &&
sbatch scripts/curation/benchmark_v3.sbatch
```

- **curation-inputs:** 4 CPUs, 32 GB RAM, up to 4 hours; no GPU. Prepares all
  YouTube/RedSage/Primus inference inputs using existing downloads and profiles.
  Allow roughly 100 GB of additional scratch storage as a planning allowance,
  including metadata and DuckDB spill; actual use depends on compression.
- **curation-v3:** 2 GPUs, 8 CPUs, 128 GB RAM, up to 2 hours. Runs the corrected
  presentation and output format on both cached Qwen models and the same 89 cases.
  This is development checking, not the full-corpus filtering run.

Both jobs use `marlowe-m000091` / `preempt`. Existing environments and model caches
are reused. No installation is needed. After job IDs are returned, disconnecting
from SSH/tmux does not cancel them.

CPU progress is logged per original source shard. The initial checksum/index
stage may be quiet while reading large files. Outputs are under
`/scratch/m000091/natedem/curation/inputs-v1/`:

- `lineage.parquet`: every valid source row and its text hash, token count and revision.
- `read_locations.parquet`: one inference read pointer per `(prompt kind, text hash)`.
- `shards/`: exact texts in bounded Parquet work units and per-input checkpoints.
- `summary.json`: final inventory of completed work units, hashes and token totals.

The first lexicographic source pointer is an I/O choice, not final source precedence.
The corpus release must still resolve all aliases, baseline/cross-kind overlap,
near duplicates and attribution. The working token counts are not additions yet.

Successful work resumes with the same code, inputs and output directory. If a job
is preempted, resubmit only that job; configuration changes require a new output
folder. Use completed manifests, not a recursive glob that could include orphans
from interrupted work. Invalid scoring responses remain review items.

## Bring back diagnostics after both finish

On the **Mac**:

```bash
mkdir -p /Users/natedemchak/Desktop/security-corpus/reports/curation/marlowe-development-v3
mkdir -p /Users/natedemchak/Desktop/security-corpus/reports/curation/inputs-v1
rsync -av \
  natedem@login.marlowe.stanford.edu:/scratch/m000091/natedem/curation/development-v3/ \
  /Users/natedemchak/Desktop/security-corpus/reports/curation/marlowe-development-v3/
rsync -av \
  --include='/summary.json' --include='/index-summary.json' \
  --include='/run-config.json' --exclude='*' \
  natedem@login.marlowe.stanford.edu:/scratch/m000091/natedem/curation/inputs-v1/ \
  /Users/natedemchak/Desktop/security-corpus/reports/curation/inputs-v1/
```

Do not copy the CPU `shards/`, `lineage.parquet`, or raw/model directories to the
Mac. If either job fails, return its log and available summary instead of
resubmitting blindly. Model comparison completion still requires checking its
coverage and semantic disagreements before production selection.

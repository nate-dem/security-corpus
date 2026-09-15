# GPU work while CPU preparation runs

The v3 reports are verified. Both models recovered 16 of the 18 assistant-eligible
cases, but admitted too many weak or damaged examples. See
[the findings](../../docs/curation_v3_review.md). This job has **no dependency on
`curation-inputs`** and reads only the existing small diagnostic packets.

## Copy on the Mac

```bash
cd /Users/natedemchak/Desktop/security-corpus
bash scripts/curation/sync_review_to_marlowe.sh
```

The additional packet has 300 complete texts: 100 from each source's existing
diagnostic pool, excluding the 89 reviewed documents and their known parent text
hashes. It contains 353,943 cl100k_base tokens. These pools have prior pipeline
exposure and are not an independent or representative corpus sample. The packet
is unlabelled; no accuracy or retained-yield claim can be made from it alone.
No raw corpus or model download is needed.

The files used by the running CPU preparation process are unchanged. Its output,
configuration, checkpoints and Python environment are preserved.

## Run on Marlowe now

```bash
cd /scratch/m000091/natedem/security-corpus
bash scripts/curation/run_review.sh --check-env &&
mkdir -p logs/curation &&
sbatch scripts/curation/review.sbatch
```

Requests **2 GPUs, 8 CPUs, 128 GB RAM, up to 2 hours** on the existing
`marlowe-m000091` / `preempt` account and partition. No Slurm dependency is set.
Leave the CPU job running; do not resubmit preparation or either old benchmark.

The GPU job performs:

1. Qwen3-8B v3 screening of the 300 additional texts.
2. A separate Qwen3-32B quality-review prompt on the original 89 cases and the
   300 additional texts. It sees source text without the 8B verdict to reduce
   anchoring. The same 32B model stays loaded between these two packets.
3. Provenance-checked comparison with the existing 8B results on the original
   89 cases, plus provisional route comparisons on the additional texts.

The original 8B/32B benchmark results are preserved; no old inference is repeated.
Agreeing eligible decisions are still candidates. Disagreements, uncertainty and
parse failures stay in review. Nothing is added to the released corpus by this job.
A second model call is not independent human validation.

Outputs are under `/scratch/m000091/natedem/curation/gpu-review-v1/`:

- `additional-screen/`, `development-critic/`, `additional-critic/`: raw attempts,
  bound prompts/configurations, labels, evidence offsets and coverage summaries.
- `*-evaluation.json`: individual run diagnostics. Additional texts have no
  reference labels, so agreement/accuracy fields are absent or null.
- `development-cascade.json`, `additional-cascade.json`: provisional comparisons;
  `selection_decision` remains null for every case.

Slurm completion means the diagnostic job finished, not that classification
accuracy passed. On preemption, resubmit this same job with unchanged inputs/code;
valid checkpoints resume. Do not rerun merely to change inconvenient judgments.

## Return results to the Mac

After this GPU job finishes:

```bash
mkdir -p /Users/natedemchak/Desktop/security-corpus/reports/curation/gpu-review-v1
rsync -av \
  natedem@login.marlowe.stanford.edu:/scratch/m000091/natedem/curation/gpu-review-v1/ \
  /Users/natedemchak/Desktop/security-corpus/reports/curation/gpu-review-v1/
```

Return the separate CPU summary using `NEXT_RUN.md` once preparation finishes.
Do not copy bulk prepared Parquet, raw shards or model weights to the Mac.

## Partition runner prepared for the following stage

`score_partitions.py` reads only explicitly requested indices from a completed
preparation manifest. It checks paths, file hashes, records, tokens and source
text bindings, and preserves all source characters. One model remains resident
across requested partitions. It emits diagnostic sidecars and resumes valid spans;
it does not implement final selection or release assembly.

Do not launch bulk scoring from this driver yet. The current task is to evaluate
the reviewer on the existing controls and additional examples, then choose the
production configuration and audit its output. Near deduplication, source
attribution, final retained counts and publication remain separate steps.

# Assistant-reviewed classifier comparison

**Current action:** v2 completed and failed development review. Use
[NEXT_RUN.md](NEXT_RUN.md) for CPU candidate preparation and the corrected v3
comparison. Commands below document the completed v2 run; do not repeat it.
See [the results](../../docs/curation_benchmark_review.md).

The user delegated review to the assistant on 2026-09-14. No manual-labeling task
is pending with the user. All 86 original development cases were assessed, then
three contrasting YouTube controls were added, including a clean forensic-triage
explanation. The current packet has **89 assistant-reviewed cases**. This is
development evidence, not independent human ground truth or a corpus-yield sample.

`rubric.py` v2 separates relevance, substance and readability, adding cited
concerns for apparent technical errors, unsupported security guarantees, and
damaged/missing content. Clear prose does not establish correct instruction.
No exhaustive fact verification is claimed.

`policy.py` routes valid labels to `eligible`, `exclude` or `review_required`.
Uncertain/unparsed results never become automatic exclusions. Relevant substantive mixed
or partially usable content stays in review for possible section selection.
Identified technical concerns block eligibility. Eligibility does not clear
licenses, attribution or duplicate handling.

`evaluate.py` verifies packet/reference bindings and supporting excerpts,
reparses raw Qwen checkpoints with request/configuration provenance, checks full
character coverage, and reports disagreements. Multi-span cases become eligible
only if every span is eligible. Individual-dimension agreement is reported only
for single-span cases because references describe complete cases.

## Run now

On the **Mac**, copy code and the small reviewed packet:

```bash
cd /Users/natedemchak/Desktop/security-corpus
bash scripts/curation/sync_to_marlowe.sh
```

On **Marlowe**:

```bash
cd /scratch/m000091/natedem/security-corpus
bash scripts/curation/run_benchmark.sh --check-env && \
mkdir -p logs/curation && \
sbatch scripts/curation/benchmark.sbatch
```

Requests **2 GPUs, 8 CPUs, 128 GB RAM, up to two hours**, using the working
`marlowe-m000091` account and `preempt` partition. The one allocation runs 8B and
32B sequentially in separate processes. 8B uses one allocated GPU; 32B uses both.
No new dependencies are required; model cache defaults to the existing
`youtube-transcripts/model-cache`. The job survives SSH/tmux disconnection.

Inputs: `reports/curation/development-v2/{packet,assistant_annotations}.json`.
Outputs: `/scratch/m000091/natedem/curation/development-v2/`:

- `qwen8b/`, `qwen32b/`: pinned configuration, all requests, raw attempts,
  reparsed labels and coverage/performance summary.
- `evaluation-qwen8b.json`, `evaluation-qwen32b.json`: disagreements with
  assistant judgments, including model-eligible/reference-not-eligible cases.

An evidence parse failure remains a review item and is included in the comparison;
it does not stop the other model from running. Operational exceptions still fail
the job. Slurm completion means comparisons finished, not that production
accuracy was established. Preempted runs resume valid spans with unchanged inputs
and code. Changes require a fresh output directory (`CURATION_OUTPUT_DIR`).

After completion, on the **Mac**:

```bash
mkdir -p /Users/natedemchak/Desktop/security-corpus/reports/curation/marlowe-development-v2
rsync -av \
  natedem@login.marlowe.stanford.edu:/scratch/m000091/natedem/curation/development-v2/ \
  /Users/natedemchak/Desktop/security-corpus/reports/curation/marlowe-development-v2/
```

These are bounded diagnostic artifacts, not bulk transcripts or model weights.
Use the comparisons to choose the full-corpus model/cascade and partitioned
production run. Production scoring, independent evaluation, near deduplication,
and final release assembly remain separate work.

## Provenance

Packet SHA-256:
`9d03cdea2cf80db4d163aa19839104212f7d17100cca9f52f04684c173c70726`.
The original packet and annotations remain in `development-v1`. Current reference
routing: 18 eligible, 43 excluded, 28 requiring review; these are case counts,
not production yield estimates. The six difficult YouTube spans resolve to four
mixed/damaged supporting-content cases and two unusable translations. None is
automatically eligible as a whole span.

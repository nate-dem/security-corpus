# Targeted recovery of v3 evidence

**Follow-up:** job 485708 completed all repair inference; 70 of 76 repairs
passed. Read the [repair review](youtube_repair_review.md) for the six remaining
scope/evidence disagreements. Do not repeat the commands below for this result;
they record the repair procedure already executed.

Reviewed 2026-09-14 after transfer of job 485673 and `pilot-qwen32b-v3`.
The GPU run processed all **637 spans / 487 documents**, with 680.3 seconds
of generation and 93.0 seconds of model loading. All responses ended normally.
**561 passed validation; 76 failed with `Substantive judgment lacks evidence`.**
Exit 2 reports incomplete evidence coverage. The shutdown nanobind messages
followed that result; they were not the recorded failure cause.

Of the 76 failures, 75 supplied six references all labeled `security_relevance`;
one supplied no references. None supplied the required `technical_substance`
reference. Fifty-two failures have raw relevance `absent`, 21 `supporting`,
and 3 `central`. An off-topic substantive explanation still needs substance
evidence under the existing rubric. Those raw labels remain unvalidated until
their evidence is complete; they are not final keep/drop decisions.

The audit reproduced original prompts from full source text and the pinned
tokenizer, request hashes, model/revision/configuration bindings, parsed
responses, and the document report. Candidate input hashes and token lengths
also passed. Full raw responses remain under `pilot-qwen32b-v3/decisions`.
Local failure inventory: `reports/youtube/review-32b-v3/failed_spans.jsonl`.

## Repair design

`repair_evidence.py` preserves the original run and creates a separate repair
sidecar. It sends **only the 76 failed focal spans**, with the original labels
as hypotheses, to the same pinned Qwen3-32B. Each request has separate arrays
for relevance, substance, and usability segment IDs. This removes the repeated
dimension-name choice that failed in the earlier response format.

The model can cite at most two segments per dimension; this is an evidence
display budget, not a document quality threshold. The same segment can be cited
for multiple dimensions. Empty arrays explicitly allow the model to report that
the source cannot support a label. Required empty evidence remains unresolved;
the parser's original requirements are not weakened.

All original classification fields, rationale, and content-form descriptions
are preserved. New evidence is explicitly marked as a separate model output,
then assembled with those unchanged fields and passed through the original
strict validator. The script does not relabel earlier citations mechanically,
fabricate quotations, or pretend the original model produced the repaired
references. The original 561 valid decisions require no new inference.

Input and repair checkpoints are bound to source decision hashes, full run
configuration, model/revision, exact prompts, and implementation hashes.
Resumption revalidates saved repair responses and skips valid ones. Original
files remain unchanged. Final selection stays null even when evidence is complete.

The actual pinned-tokenizer dry run selects 76 requests containing **304,749
input tokens**, about 15.2% of the original 2,008,315 input tokens. Its largest
prompt is 5,678 tokens, leaving room for the 768-token output budget in the
8,192 context. These are measured workload counts, not a runtime guarantee.

Local validation: 355 unit tests passed; 10 tests requiring bulk corpus data
were excluded. Ruff and Bash syntax checks passed. Tests cover unchanged
classification fields, exact evidence offsets, unsupported labels staying
unresolved, rejection of invalid IDs and mismatched provenance, preservation
of original files, and resumption without repeating successful inference.

## Historical commands for job 485708

On the **Mac**, copy the updated scripts:

```bash
rsync -av --exclude='.venv/' --exclude='__pycache__/' \
  /Users/natedemchak/Desktop/security-corpus/scripts/youtube_filter/ \
  natedem@login.marlowe.stanford.edu:/scratch/m000091/natedem/security-corpus/scripts/youtube_filter/
```

On **Marlowe**, submit the repair job:

```bash
cd /scratch/m000091/natedem/security-corpus
export YOUTUBE_DATA_DIR=/scratch/m000091/natedem/youtube-transcripts
bash scripts/youtube_filter/run.sh --check-gpu-env && \
mkdir -p logs/youtube-filter && \
sbatch scripts/youtube_filter/repair_v3.sbatch
```

This requests **2 GPUs, 8 CPUs, 128 GB RAM, up to one hour** on the working
`marlowe-m000091` / `preempt` account/partition. It uses the existing model and
environment. No new dataset, model, or package installation is required.
It should print `Preserving 561 valid spans; evidence repair targets 76 spans`.
The log is `logs/youtube-filter/youtube-evidence-v3-JOBID.log`.

Outputs go to
`/scratch/m000091/natedem/youtube-transcripts/pilot-qwen32b-v3-evidence-repair/`:

- `repairs/`: raw evidence-only attempts with timestamps and Slurm job IDs.
- `resolved_spans.jsonl`: 637 span results, identifying original versus repaired
  evidence and the SHA-256 hashes of both contributing decision artifacts.
- `document_report.jsonl`: full-document coverage and unresolved-span counts.
- `summary.json`, `run-config.json`, `requests.jsonl`: status and provenance.

After completion, run on the **Mac**:

```bash
rsync -av \
  natedem@login.marlowe.stanford.edu:/scratch/m000091/natedem/youtube-transcripts/pilot-qwen32b-v3-evidence-repair/ \
  /Users/natedemchak/Desktop/security-corpus/reports/youtube/pilot-qwen32b-v3-evidence-repair/
```

After preemption, resubmit `repair_v3.sbatch` with the same code/configuration.
If it exits 2 with unsupported original labels, review those responses rather
than repeatedly submitting the same deterministic request.

## Semantic status

The raw v3 responses now recognize the reviewed Linux `perf` instruction as
supporting and classify pharmacogenomics as absent. Disaster-modeling spans are
absent or incidental, rather than supporting. However, both spans of the biology debate `fa3519dbc86b` still
receive supporting/substantive labels, and some promotion/policy passages need
quality review. These observations are assistant diagnostics, not researcher
ground truth or measured precision/recall. Fixing citations cannot fix those
classification errors. A complete repair does not authorize a production filter
or add retained tokens; the baseline remains 1,467,709,789 reference tokens.

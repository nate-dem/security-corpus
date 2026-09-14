# YouTube 32B results and scope clarification

**Follow-up:** v3 inference and its targeted evidence repair are now complete.
See the [repair review](youtube_repair_review.md): 631/637 spans have valid
evidence, with six scope/evidence disagreements left for review. Do not resubmit
the unchanged pilot or repair. The submissions below are historical context.

Reviewed 2026-09-14. **Job 485501 was a restart that reused all 637 existing
32B decisions.** Its log contains planning and report generation, with no model
load or generation. The summary records zero attempted spans, input/output
tokens, generation seconds, and model-load seconds for that invocation. This
explains the 43-second elapsed time. Do not repeat the unchanged v2 job.

All 637 saved decisions identify `Qwen/Qwen3-32B` at revision
`9216db5781bf21249d130ec9da846c4624c16137`, with the v2 prompt. None has the
same raw response as the matching 8B decision. Their stored completion
timestamps range from `2026-09-14T15:57:48.770081+00:00` to
`2026-09-14T16:06:25.053208+00:00`. They record 1,833,777 input tokens and
106,517 output tokens. These are saved inference tokens, not new corpus tokens.
The old format did not record the originating Slurm job ID; the transferred
485501 log cannot establish which earlier job produced them or its full elapsed
time. The saved model metadata and responses are consistent with the 32B run.

## Integrity and model comparison

The audit revalidated every request using the pinned 32B tokenizer, the input
and report checksums, model/revision/prompt metadata, raw response parsing, and
exact source evidence. All 487 original texts are covered by 637 contiguous
spans with zero unresolved spans. The document report regenerates exactly.
Final selection remains null for every document. Thirty-seven documents have
an uncertain label; structural completion does not mean all labels are certain.

| Raw model relevance labels, by span | 8B v2 | 32B v2 |
|---|---:|---:|
| Central | 2 | 7 |
| Supporting | 40 | 18 |
| Incidental | 202 | 1 |
| Absent | 393 | 611 |

These counts are not precision, recall, retained yield, or an approved keep
policy. They come from a development sample combining random rows with enriched
diagnostic cases. Reviewing this sample and modifying the prompt on it cannot
establish performance on an independent held-out population.

The 32B model corrects the reviewed hemp-sustainability and SCP-fiction errors.
It recognizes AWS forensic triage and detailed cryptanalysis as direct security
instruction. Its usability labels also improve: the number of spans labeled
unusable falls from 369 to 69, including understandable off-topic material.
That change alone is not a quantitative accuracy measurement.

Remaining errors are substantive:

- `bd05cc5ee010`, characters 21576–42860: the model labels pharmacogenomics as
  supporting, while its rationale and evidence describe drug metabolism and
  clinical testing rather than computing or cybersecurity.
- `ffe6fbfc1e05`, characters 0–19953 and 19953–39249: detailed Linux `perf`,
  processor counters, cache behavior, and tracing are labeled absent because
  the text does not explicitly discuss cybersecurity. This conflicts with the
  approved inclusion of directly supporting operating-systems/software material.
- `2af84e3bf52d`, characters 21235–37493: a geopolitical discussion is labeled
  supporting even though the rationale says it lacks cybersecurity mechanisms.
- `e7bf25726928`, characters 42685–63170 and 85065–101822: disaster modeling
  and passing data/privacy concerns are stretched into supporting scope.
- `084c6e505220`, characters 0–1207: a promotional CAD/blockchain explanation
  is rated substantive without resolving whether claimed theft prevention is
  actually explained. Topic classification alone is not quality validation.

These are assistant diagnostic observations, not researcher-labeled ground
truth. Full focal texts, model labels, and evidence are preserved in
`reports/youtube/review-32b-v2/diagnostic_cases.jsonl`. The integrity audit and
comparison counts are in the same directory. `scorer-used-v2.py` preserves the
exact earlier scorer bytes for the historical checksum audit.

## Changes made

The new `rubric_scope.py` preserves the approved content scope and diagnostic
field set. It makes the label definitions explicit: actual cybersecurity
instruction is central; actual instruction in operating systems, networking,
cloud infrastructure, or software engineering is supporting even without a
security example. Other fields do not enter scope merely because they use
software or handle data. The prompt asks for the actual subject and a short
rationale before categorical labels, and for consistency with cited evidence.
This remains a proposal under test, not a proven fix or production filter.

The v1/v2 rubric files are unchanged. V3 uses the same segment evidence,
complete-text coverage, model, tokenizer, and inference budgets. It adds no
keyword filters, channel exclusions, minimum lengths, or final keep thresholds.
This development rerun must be followed by independent validation before
production selection; passing the known examples would not by itself be enough.

The scorer also fixes two implementation problems:

1. A restart now reports `inference_performed: false` when it generates nothing,
   logs the number of reused spans, and preserves cumulative attempt/token totals
   separately from this invocation. New attempts record hostname and Slurm job ID.
2. Cache reuse now checks model, revision, prompt version, full run-configuration
   hash, and span identity. Matching prompt hashes alone are insufficient because
   8B and 32B have identical tokenizer output. Missing configuration or mismatched
   provenance fails explicitly. The transferred 32B results pass their historical
   metadata audit; this gap is not evidence that they came from the wrong model.

Changed code/configuration requires a new output directory. The new job writes
`pilot-qwen32b-v3`; preserve `pilot-qwen32b-v2` and do not resubmit old job scripts
after copying the new scorer. Existing data, weights, and environments are reused.

Local verification: **344 unit tests passed**; 10 tests requiring bulk corpus
data were excluded. Ruff and Bash syntax checks passed. The actual pinned 32B
tokenizer dry run preserved all 487 documents and the same 637 span boundaries,
using 2,008,315 prompt tokens; the largest prompt is 6,243 tokens, within the
8,192-token context including the 1,024-token output allowance. These checks
validate implementation and input coverage, not v3 classification accuracy.

## Run the v3 development comparison

On the **Mac**:

```bash
rsync -av --exclude='.venv/' --exclude='__pycache__/' \
  /Users/natedemchak/Desktop/security-corpus/scripts/youtube_filter/ \
  natedem@login.marlowe.stanford.edu:/scratch/m000091/natedem/security-corpus/scripts/youtube_filter/
```

On **Marlowe**:

```bash
cd /scratch/m000091/natedem/security-corpus
export YOUTUBE_DATA_DIR=/scratch/m000091/natedem/youtube-transcripts
bash scripts/youtube_filter/run.sh --check-gpu-env && \
mkdir -p logs/youtube-filter && \
sbatch scripts/youtube_filter/pilot_32b_v3.sbatch
```

The job requests 2 GPUs, 8 CPUs, 128 GB RAM, and up to 2 hours on the existing
`marlowe-m000091` / `preempt` allocation. No package reinstall or new model is
needed. After submission it is independent of tmux. Its log is
`logs/youtube-filter/youtube-qwen32b-v3-JOBID.log`. On a fresh v3 output it should
report zero reused spans and proceed to generation; later restarts can reuse
completed v3 decisions. Reuse the same code/configuration after preemption.

When finished, transfer from the **Mac**:

```bash
rsync -av \
  natedem@login.marlowe.stanford.edu:/scratch/m000091/natedem/youtube-transcripts/pilot-qwen32b-v3/ \
  /Users/natedemchak/Desktop/security-corpus/reports/youtube/pilot-qwen32b-v3/
```

The retained baseline remains **1,467,709,789 cl100k_base tokens**. These pilots
have not added final tokens. The web pool and 3B budget are documented in
[the expansion inventory](expansion_results_review.md).

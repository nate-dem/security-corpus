# Corpus recovery review — 2026-09-08

The architecture is suitable for continued pretraining: assembled plain-text
documents, queryable source-family features, immutable normalization, exact
hashes, and separate semantic-decision sidecars. The recovery checkpoint is
usable for continued work. It is not yet a release candidate, and neither
classifier effectiveness nor downstream model improvement has been established.

**Scope correction (2026-09-09):** the large recovery checkpoint audited below
is an upstream working set, not the previously cleaned and filtered selection.
Existing retained outputs were subsequently checked and are documented here.
The initial recommendation to begin another Qwen run was premature. Preserve
the existing selection, reconcile its component files, and validate that
candidate before deciding whether any specific records need reprocessing.
Missing immutable model-revision metadata limits reproducibility; it does not
by itself demonstrate that existing filtering decisions are wrong.

## Existing cleaned and filtered outputs

| Component | Existing path | Rows | Stored tokens |
|---|---|---:|---:|
| Qwen-retained Q&A | `data/filtering/v3/qwen_qa_kept/` | 217,279 | 284,619,801 |
| Cleaned seed papers | `data/final-no-chunk/academic_papers/arxiv_cs_cr_full.parquet` | 46,273 | 661,806,017 |
| Cleaned retained citation papers | `data/final-no-chunk/academic_papers/arxiv_citation_qwen_kept_full.parquet` | 17,067 | 247,649,077 |
| CloudTrail sessions | `data/final-no-chunk/normalized/source_id=cloudtrail-flaws/part-00000.parquet` | 56,043 | 246,135,664 |
| Sigma rules | `data/final-no-chunk/normalized/source_id=sigma/part-00000.parquet` | 3,706 | 1,678,011 |

These counts were read from the actual files. All four `final-no-chunk` files
match that directory's manifest counts and token totals. Every retained Q&A
row matches a successful keep decision in
`data/filtering/v3/qwen_qa_decisions.parquet` on source, record ID, and content
hash. Other cleaned structured sources exist under
`data/training-clean-v2/normalized/`. The retained components span directories;
a consolidated, publish-ready all-source candidate has not yet been verified.
The checks above establish artifact presence and selection consistency, not a
new full content-quality assessment or token recomputation.

Including the five cleaned structured sources (NVD, KEV, ATT&CK, CWE, CAPEC),
the retained baseline totals **659,147 records and 1,467,709,789 stored
`cl100k_base` tokens**, excluding YouTube. The local per-component count report
is `reports/recovery/review-2026-09-09/retained_token_inventory.json`.
The researcher's current direction is to carry this baseline forward and
proceed to YouTube ingestion and filtering.

The historical training-clean-v2 manifest lists 4,545 YouTube transcript rows,
but no YouTube artifacts were found under the current local `data/` tree.
Recovering or collecting those transcripts remains the next source task.

This review started at commit `dcef7ed`. It inspected recovery construction,
Qwen prompts/scoring/restarts/merging, arXiv downloading and extraction, release
audits, source-license policy, and Marlowe job templates. Existing connector
behavior was exercised by the fixture suite. This was not a re-ingestion of
every upstream dump or a model-training evaluation.

## Evidence from the recovered files

A complete local hash/structure pass read all 37 selected Parquet files in the
QA universe, structured checkpoint, and two academic full-text checkpoints.
The machine-readable evidence is local at
`reports/recovery/review-2026-09-09/integrity.json`; it records exact file hashes,
schemas, source totals, and bounded examples. Corpus text remains outside Git.

| Check | Result |
|---|---:|
| Rows examined, before academic exact dedup | 2,089,280 |
| Stored reference-token total | 2,703,285,484 |
| Recomputed content-hash mismatches | 0 |
| Duplicate `(source_id, record_id)` rows | 0 |
| Rows missing provenance URL | 56,043, all CloudTrail |
| Exact-content duplicate groups | 4,961 |
| Extra exact-content copies | 28,360 |
| Cross-source exact-content groups | 6 |

The README's 2,089,231 documents and 2,703,091,696 tokens account for 49 academic
duplicate copies. Structured duplicates remain included by design. These are
stored token totals: this review did not recompute all 2.7 billion tokens. The
new integrity command supports that CPU job and explicitly labels a skipped
token check as insufficient for release integrity.

Real prompts rendered successfully from all 28 QA source partitions. The
baseline suite had 236 passing tests; the added regressions cover the failures
below. Local CPU checks do not establish vLLM/CUDA compatibility, GPU throughput,
or semantic filtering accuracy. The optional one-GPU smoke job exercises
inference compatibility and output validity only if a new scoring run is needed.

## Implemented repairs

| Finding | Repair and effect |
|---|---|
| Resuming Qwen skipped parse failures permanently | Only successful, decided rows are skipped; failed parses are eligible for another attempt. |
| Retrying a subset of an array changed its hash modulus | QA retains 100 shards and citations retain 24, regardless of the submitted index subset. |
| Merging could select a failed retry over success or hide conflicting decisions | Successful attempts take precedence; conflicting successful keep/drop labels and mixed configurations stop the merge. |
| A resumed run could reuse unprovenanced outputs or changed code/runtime | A run config is required for existing parts; scorer/prompt hashes, source selection, inference settings, and runtime versions are checked. |
| SQL NULLs bypassed some coverage/selection gates | Shared validation rejects null/invalid provenance, mixed tasks/prompts/models, missing decisions, and non-Boolean decisions; citation selection binds the expected abstract task and prompt. |
| Fractional/nonfinite scores could be truncated or crash parsing | Invalid score types produce explicit parse failures. |
| Oversized prompts failed without identifying the document | Actual model-token length is checked with the output budget and record ID; no extra truncation policy is introduced. |
| Rebuilding QA/structured/citation checkpoints deleted prior outputs before success | Builds stage first, then replace outputs with rollback for ordinary publication errors. Multiple renames are not power-loss atomic. |
| LaTeX includes inside comments/code were expanded, and repeated includes looked circular | `latex-v3` expands active TeX only and tracks the current recursion ancestry. |
| A rejected compressed tar could fall through to the gzip-as-TeX path | Archive rejection now stops extraction; no fallback reinterprets rejected tar bytes. |
| Missing status admitted legacy papers; stale LaTeX could override PDF output | A supported completed status is required and determines which content file is read. |
| Download interruption left files that looked complete | Downloads stage before replacement, reject non-source responses, and retain failures for retry. |
| Metadata failures were checkpointed as completed, ahead of buffered data | Durable parseable JSONL determines completion; successful writes are flushed before checkpointing and failures retry. |
| A changed archive or missing output could be skipped by normalization | Resume verifies input SHA-256, normalizer version, and content-file existence; PDF writes are atomic. |
| Older selected papers could re-enter from the cache | arXiv ingestion accepts an explicit ID list for the current seed-plus-accepted set. |
| License audit silently ignored nonexistent inputs and unrecognized policy states | Requested paths must exist; unknown states fail closed. |
| Final hash/token recomputation was a documented gate without a general executable check | Added a streaming, multiprocess integrity audit with file checksums, source totals, and duplicate reporting. |
| QA requested 12 hours on a four-hour preempt partition | Template now requests four hours and defaults to four concurrent jobs. |

No source selection, semantic prompt policy, numeric quality threshold, schema
field, or license permission was invented. Existing recovered data files were
left intact. GitHub Actions now runs lint, unit tests, and shell syntax checks.

## Checks to reconcile against the retained candidate

The original review below assessed the larger recovery working set. Its counts
and proposed reruns must not be treated as evidence of defects in the retained
selection. First establish which existing cleaned files form the intended
candidate, then apply the relevant checks to those files.

1. **Validate the classifier policy if another run is needed.** Qwen's system prompt
   prefers dropping uncertain/basic material, while the citation prompt asks
   to keep borderline adjacent research. That tension is a research choice.
   The QA prompt also sees community scores and only the head/tail of long
   threads. Review the exact rendered prompts and both keep/drop strata;
   document sample size, reviewers, and the acceptance decision. The human-label
   checker validates supplied rows, but does not yet prove they are the original
   sample or bind approval to final release bytes. Retain the original sample
   manifest and require explicit signoff at assembly.
2. **Validate the existing cleaned papers and their provenance.** The cleaned
   46,273 seed papers and 17,067 retained citation papers already exist in
   `data/final-no-chunk/`. Inspect these outputs and reconcile their extraction
   history before proposing replacements. Preserved source statuses are
   unversioned, so provenance needs further verification. If specific papers
   require re-extraction, record each selected ID as emitted or an explained
   failure and preserve raw archives and diagnostics. Changing the citation
   selection requires a researcher decision. Archive hashes identify exact input
   bytes, but a mutable `/src/{id}` request plus separately harvested metadata
   does not itself prove that a paper version and its license match. Bind
   version-specific metadata before public full-text release. LaTeX macro and
   conditional expansion and PDF reading order also require content sampling.
3. **Approve contribution-level attribution for Stack Exchange.** Current
   connectors compute contribution license expressions, but do not retain
   question/answer author identifiers and contribution lineage. Re-ingestion
   alone with the current connector does not repair that gap. A proposed
   contribution sidecar would retain post ID, role, author identity/profile URL,
   contribution URL, revision/license evidence, and corpus record ID. Approve
   that representation before implementing it. The license policy must then
   recognize reviewed composite license expressions and attribution evidence.
4. **Resolve long records and duplicates.** The structured audit reports
   CloudTrail documents up to 8,685,255 tokens and a Sigma rule up to 137,139.
   Choose the training context budget and treatment of those records, then
   implement lossless chunking with lineage if selected. Document precedence
   for the 28,311 structured duplicate copies. The existing QA canonical rule
   prefers a prior exact-key decision, then source/record ID; confirm that it
   remains the intended source/metadata precedence for release.
5. **Resolve source policies for the candidate.** The upstream NVD checkpoint
   includes 16,529 Rejected records (682,974 stored tokens); this does not show
   that the cleaned NVD selection retains them. Check the selected files before
   reopening that filtering decision. Under the existing release
   policy, Reddit and CloudTrail remain blocked pending permission/provenance;
   restricted arXiv licenses require source-specific treatment. Conditional
   license-audit results are not proof that required notices/attribution were
   packaged. Add that evidence to the candidate before publication.
6. **Define and assemble v1.** Record target capabilities and an evaluation
   boundary, source set, dedup policy, context handling, and split/contamination
   policy. The repository still needs a candidate assembler, a release manifest
   binding every gate/signoff to exact files, and a Hugging Face data card.
   Choose the repository's code license. Near-duplicate handling remains a
   downstream decision; this checkpoint proves exact hashes only.

The writer currently materializes a whole source before Arrow conversion, and
each Qwen shard scans the full input. These are scaling limits to measure in
the pilot, especially for paper ingestion and shared-storage traffic. They do
not justify changing corpus policy. The pinned GPU environment and model
revision still need resolution and execution on Marlowe; no GPU job was run in
this review.

## Optional compute preflight

Do not submit a Qwen rerun solely because of this review. If candidate
validation establishes a need for new scoring, start with
`scripts/classify/slurm/qwen_smoke.sbatch`: one H100, eight CPUs,
64 GiB host RAM, one hour. It selects the shortest and longest QA record per
source plus two citation abstracts, then verifies decision coverage. This is
an implementation smoke test, not a representative human-quality sample.

The script writes separate sample/decision files and does not replace the
existing cleaned corpus. It does not collect YouTube transcripts.

For an approved scoring rerun, the existing arrays request one GPU per shard,
with four concurrent jobs. Full token recomputation and paper extraction need
CPUs, not GPUs. The local Marlowe runbook contains exact commands and remains
excluded from Git. The immediate work is to reconcile the existing cleaned
candidate and recover/add the missing YouTube transcripts; no GPU allocation
is currently required for that reconciliation.

Primary references checked during review:

- [Marlowe scheduling](https://marlowe-research.stanford.edu/documentation/slurm/)
  documents the four-hour preempt limit and allocation accounting.
- [Marlowe hardware](https://docs.marlowe.stanford.edu/specs) documents H100 80GB GPUs.
- [Slurm array variables](https://slurm.schedmd.com/job_array.html) explains why
  array task count changes on subset retries.
- [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B) documents non-thinking
  mode and vLLM support. This does not validate classifier quality for this corpus.
- [Stack Overflow licensing](https://stackoverflow.com/help/licensing) ties
  contribution licenses to dates/revisions.
- [arXiv licensing](https://info.arxiv.org/help/license/index.html) distinguishes
  paper licenses and notes that versions may have different licenses.

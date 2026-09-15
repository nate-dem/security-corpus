# Corpus completion status — 2026-09-15

**Current priority: research quality and publishability.** The researcher
clarified that 3B is aspirational, not required. CPU jobs 485980 and 485981
completed, and their transferred reports were checked against the producing
code, configuration, and aggregate counts. English preparation yielded
7,312,515,096 exact-unique candidate tokens; the baseline was independently
recounted at 1,467,709,789 tokens. See [the report review](expansion_preflight_review.md)
for the remaining integrity and publication findings.
See [the current commands and curation protocol](curation_protocol.md).
The completed download/profile jobs and old evidence repair should not be repeated.

**Immediate next job:** [single-GPU source quality recheck](../scripts/curation/QUALITY_PASS.md).
Job 486937 completed both models, and all ten evaluations were re-parsed locally.
The 27B model recovered more useful controls but still admitted damaged text;
two output-limit failures remain unresolved. The next pass reviews 1,015 full
documents with the cached 27B model and a focused source-only critic. No download
or environment reinstall is needed. See [the findings and compute estimate](first_batch_review.md).
CPU preparation and GPU quality review completed. The prepared inventory contains
19,310,285 work-unit texts / 21,020,784,191 candidate tokens. The old 8B/32B pair
still accepts weak or damaged content; assistant review of 48 jointly accepted
examples left 20 eligible candidates, 20 needing review and eight excluded.
The official Qwen3.6-35B-A3B-FP8 and Qwen3.8-27B-FP8 checkpoints have now been
compared on the same 137 controls and 4,187 complete candidate documents.
See [the findings](curation_gpu_review.md). No user labeling task is pending.
Assistant references are not independent human evidence.

## Completed work and current counts

| Workstream | Completed | Still missing |
|---|---|---|
| Existing baseline | 659,147 records, 1,467,709,789 recomputed cl100k_base tokens; stored hashes and token counts match | Resolve two ATT&CK ID collisions and missing CloudTrail provenance; final release assembly and verification |
| YouTube | All 439 shards prepared; 3,262,750 English rows, including translations; 2,944,120 exact-unique nonblank texts / 7,312,515,096 candidate tokens | Validate production filtering and produce retained output |
| YouTube classifier | Full inputs prepared; newer-model comparison completed and reviewed | Validate second-pass quality, production configuration and selection |
| RedSage and Primus | All 1,724 files prepared; exact overlap measured; development cases and newer-model comparison reviewed | Second-pass quality assessment, independent evaluation, scalable production filtering, near deduplication, retained output |
| Release | Baseline preflight complete; draft manifest and dataset card exist | Resolve reported source-policy mismatches and permission/attribution gaps, consolidate final Parquet, validate and publish |
| GitHub | Implementation checkpoints on codex/corpus-recovery-20260826; see Git history for exact revisions | Commit subsequent changes incrementally; code license remains a researcher choice |

The two web sources supply **13,708,013,960 novel exact-unique candidate tokens**
after comparison with the baseline. Reaching the aspirational 3B target would
require another **1,532,290,211 tokens**, or about 11.18% of that web pool before
any YouTube contribution. This is arithmetic, not a yield prediction or a reason
to admit weak material. A smaller corpus can meet the release objective.

No final YouTube or web additions have been counted. The 7.313B YouTube tokens
are candidates before semantic filtering and near deduplication. Exact overlap
with the baseline and web additions has not yet been subtracted from that total.
The current baseline is a retained working corpus, not yet a verified public release.

## Recommended order to finish

1. **Finish classifier validation and the missing production code.** Use saved
   examples to address the observed relevance/substance/usability errors. Build
   a web-specific calibration workflow using the already transferred samples;
   the transcript prompt cannot simply serve as validation for web pages.
   Separate development examples from independently reviewed evaluation data.
   Existing language and supporting-topic approvals remain in force. Present
   unresolved boundaries and final selection rules concretely for researcher
   review. Full preparation and resumable partition scoring are implemented;
   production selection and release assembly are still missing.
2. **Run the validated expansion workflow on Marlowe.** Compare the sources by
   quality and complementary contribution. CPU jobs prepare and deduplicate candidates; GPU jobs classify;
   CPU jobs apply the reviewed policy. Measure throughput and retention before
   choosing the production array size. The current comparison uses one GPU per
   model and preserves old environments in a separate runtime.
3. **Assemble and measure the retained corpus.** Preserve source/variant lineage,
   handle exact and near duplicates across additions and baseline, retain context
   in any selected transcript passages, and compute actual output token counts.
   Report the actual size after all removals. Do not lower the quality bar or
   count repeated copies to meet the aspirational target.
4. **Complete release packaging.** Resolve outstanding per-source attribution
   and redistribution treatment, finalize the output schema and manifest, audit
   the exact release files, and write the Hugging Face dataset card. Publicly
   releasable scope can affect the final count and should be reviewed alongside
   filtering, rather than discovered after assembly.
5. **Commit and publish.** Review and commit code incrementally while the large
   jobs run; keep bulk data and credentials outside Git. Push the tested code,
   upload the approved release to Hugging Face, verify the uploaded inventory,
   and retain a durable copy of manifests and decisions off scratch.

Steps 1 and release-policy preparation use current local artifacts. Do not
resubmit the completed English preparation or baseline preflight jobs.
Assistant review and the v3/critic comparison reviews are complete. The separate
environment and two pinned FP8 models are installed on Marlowe and working. Run
only the single-GPU quality recheck next. Production scoring at scale and
near-dedup assembly remain unfinished.

## Evidence and detailed runbooks

- [Expansion inventory and sample findings](expansion_results_review.md)
- [Latest YouTube repair and semantic review](youtube_repair_review.md)
- [Retained baseline reconciliation](recovery_review.md)
- [Source publication policies](source_licenses.md)

Older job commands elsewhere in the repository document completed experiments.
Use the current-status paragraph at the top of each runbook before submitting.

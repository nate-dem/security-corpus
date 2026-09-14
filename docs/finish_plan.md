# Corpus completion status — 2026-09-14

**Current priority: research quality and publishability.** The researcher
clarified that 3B is aspirational, not required. CPU jobs 485980 and 485981
completed, and their transferred reports were checked against the producing
code, configuration, and aggregate counts. English preparation yielded
7,312,515,096 exact-unique candidate tokens; the baseline was independently
recounted at 1,467,709,789 tokens. See [the report review](expansion_preflight_review.md)
for the remaining integrity and publication findings.
See [the current commands and curation protocol](curation_protocol.md).
The completed download/profile jobs and old evidence repair should not be repeated.

## Completed work and current counts

| Workstream | Completed | Still missing |
|---|---|---|
| Existing baseline | 659,147 records, 1,467,709,789 recomputed cl100k_base tokens; stored hashes and token counts match | Resolve two ATT&CK ID collisions and missing CloudTrail provenance; final release assembly and verification |
| YouTube | All 439 shards prepared; 3,262,750 English rows, including translations; 2,944,120 exact-unique nonblank texts / 7,312,515,096 candidate tokens | Validate production filtering and produce retained output |
| YouTube classifier | 487-text / 637-span pilot; 631 evidence-valid spans after repair | Reviewed accuracy evaluation, resolution of scope/quality disagreements, scalable production runner and selection |
| RedSage and Primus | All 1,724 files downloaded and profiled; exact overlap measured; development review packet and source-specific scorer implemented | Human review and independent evaluation, production filtering, near deduplication, retained output |
| Release | Baseline preflight complete; draft manifest and dataset card exist | Resolve reported source-policy mismatches and permission/attribution gaps, consolidate final Parquet, validate and publish |
| GitHub | Implementation checkpoint pushed on codex/corpus-recovery-20260826 at 24c3d16 | Commit subsequent changes incrementally; code license remains a researcher choice |

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
   review. Full English preparation is complete. Build resumable, partitioned production
   scoring; the current `prepare.py` and `score.py` are bounded pilot tools.
2. **Run the validated expansion workflow on Marlowe.** Compare the sources by
   quality and complementary contribution. CPU jobs prepare and deduplicate candidates; GPU jobs classify;
   CPU jobs apply the reviewed policy. Measure throughput and retention before
   choosing the production array size. No new model or GPU allocation is chosen
   in this status update.
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
GPU development evaluation follows review of the new packet; production scoring
and near-dedup assembly remain unfinished. No dependency reinstall or download
is needed for the current review.

## Evidence and detailed runbooks

- [Expansion inventory and sample findings](expansion_results_review.md)
- [Latest YouTube repair and semantic review](youtube_repair_review.md)
- [Retained baseline reconciliation](recovery_review.md)
- [Source publication policies](source_licenses.md)

Older job commands elsewhere in the repository document completed experiments.
Use the current-status paragraph at the top of each runbook before submitting.

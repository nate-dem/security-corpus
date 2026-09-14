# Review of completed CPU jobs — 2026-09-14

**Update:** the user delegated the manual-review work to the assistant. The
original examples are assessed and a revised 89-case comparison is ready. Use
[the current commands](../scripts/curation/README.md); the manual steps below
remain as historical instructions, not a pending task for the user.

Jobs 485980 (English preparation) and 485981 (baseline preflight) completed.
Transferred JSON reports were checked locally against their manifest/configuration
hashes, producing Python modules, tokenizer identity, prior profile, and count
arithmetic. All checks passed. This validates the reports' consistency; the
large English Parquet outputs remain on Marlowe and were not rehashed locally.

## Counts

| Inventory | Records/texts | cl100k_base tokens | Interpretation |
|---|---:|---:|---|
| Retained baseline | 659,147 | 1,467,709,789 | Full recomputation agrees with stored lengths |
| English nonblank rows | 3,218,693 | 7,456,901,154 | Includes repeated text and translations |
| English exact-unique texts | 2,944,120 | 7,312,515,096 | Unfiltered candidates; not retained additions |
| Web additions after exact overlap with baseline | 16,364,317 | 13,708,013,960 | Earlier comparison; still unfiltered candidates |

English preparation accounts for all 439 raw shards and all 3,262,750 English
rows. There are 44,057 blank rows and no missing-text rows. Exact deduplication
identifies 274,573 extra copies, accounting for 144,386,058 repeated tokens.
YouTube's overlap with the baseline/web pool and all near-duplicate overlap
remain unmeasured. These candidate counts must not be summed into a claimed
retained corpus size. No final expansion tokens have been added yet.

## Baseline findings

All stored content hashes and token counts matched recomputation. There were
no exact-content duplicate groups. Integrity nevertheless fails for:

- Two duplicate record-ID groups: `mitre-attack:G0007` (APT28) and
  `mitre-attack:G0088` (TEMP.Veles). Each occurs twice with different content.
  Local ATT&CK file checksums match the Marlowe manifest. Inspection identifies
  different revisions of the same STIX objects, rather than unrelated objects
  sharing a human-readable name. Final assembly needs explicit version
  precedence with both input variants retained in provenance. No rows were
  removed or relabeled during this review.
- All 56,043 CloudTrail rows have a null `source_url`. This is a missing
  provenance field, not evidence that event content is corrupted. Establish the
  actual archived log source before supplying a value; do not fabricate a
  per-session URL from the session ID.

The existing source-publication policy also reports unresolved terms and
attribution. Its 436,582 `unknown` records include historical license strings
that do not match the current policy (NVD, CISA, Sigma, CloudTrail, and Reddit).
The label `unknown` is not a legal conclusion and must not be fixed by blindly
renaming fields. Other findings concern contribution attribution for Q&A,
paper-level arXiv terms, and missing source permissions. Only 279,373,668 tokens
currently land in the policy's `conditional` category; even those still require
their notices and attribution. This is not a finalized public-subset count or
a recommendation to discard the other working data.

## Immediate next action

The local development packet is ready at
`reports/curation/development-v1/review.html`. Open it on the Mac:

```bash
open /Users/natedemchak/Desktop/security-corpus/reports/curation/development-v1/review.html
```

Use the case selector to start with cases **81–86**, the six unresolved YouTube
spans. Cases 1–40 are RedSage web examples and 41–80 are Primus web examples.
Judge relevance, substance, usability, language, and mixed content separately;
enter a supporting passage/reason and click **Mark reviewed**. Use `uncertain`
when the text cannot support a judgment. Export annotations before closing.
Partial exports are useful for development and do not establish validation
coverage or authorize production selection.

Save the export as
`reports/curation/development-v1/annotations.json`. The exported packet/text
hashes allow labels to be checked against exactly what was reviewed. These are
development cases; a separate evaluation sample is still required after the
rubric is settled. No GPU allocation is needed for this step.

The human judgment is needed because past classifiers produced evidence-valid
but incorrect topic/substance labels. Another pass that merely fills empty
evidence arrays does not establish filtering quality. The repository assigns
final filtering decisions to the researcher; assistant diagnostics remain
distinct from human reference labels.

## Report identities

- English preparation config SHA-256:
  `77bd22213314bae6b0308f144e59d816feaa208f992ae6420e96b4521719537d`
- English unique-index SHA-256 recorded by Marlowe:
  `68c7044057e5ce2c4641d42202f8565e7365a11c5a75028256be3e7322046e92`
- Baseline draft-manifest SHA-256:
  `afa9f4055145fb25fd4bf5f84f17f77fddc72e63a231e1250dd27593f9a6f7f0`
- Development review packet SHA-256:
  `3a029b4f160aed326fb93b1e8f492ad89529d9946699002786f5b9e8727c23f2`

# Security corpus — dataset card draft

**Status: development; no release artifacts are declared ready by this card.**
Replace this draft with a card generated against the final manifest before upload.

## Purpose

A corpus of plain-text documents for continued pretraining on cybersecurity and
directly supporting operating systems, networking, cloud infrastructure, and
software engineering. It is not an instruction-tuning or chat dataset. Research
quality and publishability take priority over volume. Three billion cl100k_base
tokens is an aspiration; a smaller release is acceptable.

## Data and provenance

The current working baseline has 659,147 retained documents and 1,467,709,789
stored reference tokens. These are not final release counts. Candidate additions
are English YouTube-Commons transcripts, including translations, and qualifying
RedSage-CFW and Primus-FineWeb records. Final source inclusion is pending.

For each released artifact, publish a manifest with exact SHA-256 checksums,
record and recomputed token counts, source revisions, and the processing code
commit. Preserve document and source identifiers, URLs, applicable licenses,
attribution, and the lineage of any extraction or segment selection. Do not
replace an upstream page's missing terms with its host dataset's license.

## Curation and quality evaluation

The existing baseline retains its prior selection and documented provenance
limitations. Additions are assessed from actual text using separate relevance,
technical substance, usability, and mixed-content judgments. Model citations
are checked against source text; this establishes evidence location, not truth
or classification accuracy. Model outputs remain distinct from human labels.

Publish before release:

- The final rubric, model and tokenizer revisions, sampling and selection policy.
- Independent evaluation design, reviewer instructions, agreement, and observed
  errors on predicted keeps and drops. Development examples are not held-out tests.
- Counts through structural checks, relevance/quality selection, uncertainty,
  exact and near deduplication, and any publication exclusions, by source.
- Treatment of mixed/long documents and evidence that selected segments retain
  enough context without counting overlapping text as additional unique tokens.

No downstream training benefit, precision/recall, or corpus-wide factual
verification has yet been established for the additions.

## Limitations and appropriate use

Expect source and sampling biases, outdated technical claims, and errors in
transcription, translation, and extraction. Security content can describe both
offensive and defensive techniques. A usable explanation is not proof of factual
correctness. Document review of sensitive personal information and the handling
of published examples of credentials or identifiers. Do not equate a pattern
match with a real secret or silently damage code examples through redaction.

Record any benchmark-overlap checks and their exact benchmark versions. Do not
claim contamination-free evaluation without those checks. Corpus publication
does not establish performance improvements in any trained model.

## Terms, attribution, and corrections

Per-source and per-record terms apply; no blanket full-text license is assigned
in this draft. Package notices and attribution for the exact released subset.
Sources requiring additional permission or evidence need their release treatment
resolved separately from internal retention. See `docs/source_licenses.md` and
`config/source_licenses.yaml` for the existing engineering policy.

Before publication, specify a maintained contact or issue channel for corrections,
attribution requests, and removal requests, and document the versioning process.

## Release inventory and citation

Pending final manifest: source/config names, record counts, token counts, storage
size, processing commit, release version, upload revision, and citation. These
values must come from verified release artifacts rather than this draft.

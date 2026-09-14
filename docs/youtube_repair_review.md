# YouTube evidence repair review

Reviewed 2026-09-14 after transfer of job **485708**. **Do not resubmit the
unchanged repair job.** It generated all 76 requested responses; 70 repairs
passed, and six explicitly supplied no evidence for the original supporting
relevance label. Exit 2 records those unresolved judgments.

| Result | Spans |
|---|---:|
| Original valid evidence, preserved | 561 |
| Newly repaired evidence | 70 |
| Total with valid evidence | 631 / 637 |
| Still missing relevance evidence | 6, across 4 of 487 transcripts |

All six remaining responses are complete, parseable JSON. They supply substance
citations and an explanation, but leave `security_relevance` empty. The original
assessment of each was `supporting`, `substantive`, `usable`, and not mixed.
The validator correctly refuses to treat these as fully supported assessments.
This is a disagreement between model judgments, rather than missing inference
or a broken transfer. The six remain unresolved in the saved outputs.

## Audit and implementation

The new `scripts/youtube_filter/audit_repair.py` reproduces all 637 original
requests from the source text and pinned tokenizer, all 76 repair requests,
checkpoint provenance, strict response parsing, merged results, and document
coverage. It checks that repair did not change any original classification
field. It requires the recorded parser implementations and an already cached
tokenizer; it does not load weights or invoke inference.

The transferred artifacts passed this audit. Every saved repair attempt records
Slurm job 485708. Local outputs are:

- `reports/youtube/repair-review-v1/audit.json`: reproducible integrity and
  coverage findings, including a checksum for the review packet.
- `reports/youtube/repair-review-v1/review_required.jsonl`: the six unresolved
  spans, full focal text, original assessment, repair explanation, and lineage.
- `reports/youtube/repair-review-v1/assistant_review.jsonl`: diagnostic readings
  below with exact source excerpts and character offsets, kept separate from
  model outputs and final selection.

The source repair summary SHA-256 is
`db7e821da769f275d307c5a8b7d550fb51ce85746fda1d985bb378131dfcae1e`.
Audit success certifies artifact consistency, not classification accuracy.
Validation for this addition: **355 unit tests passed**, with 10 bulk-data tests
excluded. Ruff passed. Tests cover both complete and unresolved repair audits,
unchanged original judgments, and rejection of corrupted merged outputs.

## What the six cases show

These are assistant diagnostics under the already approved security plus
supporting computing scope, not researcher ground truth or keep/drop decisions.
IDs below abbreviate the candidate content hash; ranges are zero-based,
end-exclusive character offsets in the complete transcript. Segment IDs are
local to each focal span.

| Candidate and range | Source evidence | Diagnostic finding |
|---|---|---|
| `61b5963fe8a9`, 19670–38798 | Segments 32–33 explain a Python decorator/render loop and particle updates; 43–44 discuss moving work off the JavaScript UI thread. | Actual programming instruction is present despite the repair's claim otherwise. The span also mixes art, music, and chatter; the original `mixed_content: no` needs review. |
| `61b5963fe8a9`, 38798–58905 | Segments 12, 14, and 16 explain a rhythm DSL, its representation, and a visual mapping rule. | A supporting programming passage exists within a mixed presentation. Requiring cybersecurity or OS instruction overlooks the approved software-engineering scope. Missing demonstrations and unrelated discussion need separate quality review. |
| `6d52ef73a8f9`, 27003–45000 | Segments 7, 10, and 13–14 compare i3/bspwm workspace behavior and Python-based window-manager configuration. | This is supporting OS configuration material. The repair itself describes the mechanism, then withholds relevance for lack of security content. Translated wording and livestream chatter remain separate concerns. |
| `d38271a5d9d3`, 18404–36963 | Segments 8–9 discuss equality/inequality defaults and recursive definitions; 31 identifies a type-class method. | The programming topic qualifies as supporting. Garbled translation and missing on-screen code make the original `usable` judgment questionable. A relevant topic alone does not establish usable training text. |
| `d38271a5d9d3`, 36963–55146 | Segments 13 and 16 discuss data constructors, product types, and tuples. | Again, supporting programming is present. Translation quality needs independent assessment; the repair's explanation overstates how clearly the text teaches the concepts. |
| `ea95f8c81d4b4`, 18481–37082 | Segments 1 and 6 describe Blender Geometry Nodes operations; 26–27 describe a point-cloud import workflow. | This is a boundary between visual programming instruction and using an application for artwork. It also depends on unseen demonstrations. Leave it for scope/quality adjudication rather than inferring a blanket rule for creative software. |

Five spans therefore contain identifiable supporting computing material, even
though their complete text may not pass a later quality or mixed-content rule.
The sixth is a scope boundary. None requires another identical evidence request
to understand the disagreement.

## Classification quality remains a separate issue

Among the 631 evidence-valid spans, the model assigns 599 absent, 5 incidental,
23 supporting, and 4 central labels. These are descriptive counts, not accuracy
or corpus-yield estimates. The pilot combines a random arm with enriched
diagnostic cases and must retain that sampling distinction.

Known errors also exist outside the six failures. Both spans of the biology
debate `fa3519dbc86b` have supporting/substantive labels despite discussing a
different scientific field. The CAD/blockchain promotion `084c6e505220` needs
substance review: a promised IP-protection capability is not itself a technical
explanation. These cases demonstrate why making every response pass the parser
would not establish a reliable production filter.

The next evaluation should separate relevance, substance, usability, and mixed
content in a reviewed calibration set, retain all judgments from each model
version, and measure performance on an independently reviewed random sample.
The five clear supporting examples above do not require a new language or
subject-scope decision. Borderline creative workflows and final quality/selection
rules still need researcher adjudication. No production threshold is introduced
by this review.

No Marlowe job is needed to review these saved outputs. The retained corpus
remains **1,467,709,789 cl100k_base tokens**; this pilot has added no final
selected tokens.

## Reproduce the local audit

Already run successfully on the Mac using the cached 32B tokenizer:

```bash
cd /Users/natedemchak/Desktop/security-corpus
HF_HUB_CACHE=/tmp/security-qwen32-tokenizer PYTHONPATH=src:. \
venv/bin/python -m scripts.youtube_filter.audit_repair \
  --input-dir reports/youtube/pilot-v1 \
  --source-dir reports/youtube/pilot-qwen32b-v3 \
  --repair-dir reports/youtube/pilot-qwen32b-v3-evidence-repair \
  --output-dir reports/youtube/repair-review-v1
```

Exit 0 means the audit completed; `evidence_coverage_complete: false` remains
the expected result for this run. The command fails if the tokenizer cache is
unavailable, rather than downloading model assets. It writes only its separate
review outputs.

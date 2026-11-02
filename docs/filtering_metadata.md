# CyberOpen filtering metadata

The [published tables and field guide](https://huggingface.co/datasets/cyberopen-research/CyberOpen/tree/main/metadata/filtering) accompany the dataset. The [selection policy](../config/filtering_selection.json) is also included in this code repository.

These tables contain the original automated filtering assessments for retained
Primus-FineWeb, RedSage-CFW and YouTube documents. They cover 815,460 documents:
536,414 Primus-FineWeb, 184,930 RedSage-CFW and 94,116 YouTube transcripts.
The transcript assessments cover 138,208 passages.

The tables are research metadata, separate from the training documents. The
default dataset configuration continues to load only `data/*/*.parquet`.
All included scores, labels, explanations and identifiers retain their original
values. No document was rescored or reselected for this supplement.

## Files and fields

| File | Rows | Relationship to training documents |
|---|---:|---|
| `web-records.parquet` | 721,344 | One row per retained web document |
| `youtube-records.parquet` | 94,116 | One row per retained transcript |
| `youtube-spans.parquet` | 138,208 | One or more rows per retained transcript |

Join to the training corpus using both `record_id` and `content_hash`. The hash
is SHA-256 of the UTF-8 document text. Check that it matches before applying the
assessment to a document. Joining the passage table directly to training rows
creates repeated documents; aggregate passages first when document-level output
is needed.

Common document fields are `record_id`, `content_hash`, `source` and
`content_length`. Here, `source` corresponds to the training field `source_id`,
and `content_length` counts `cl100k_base` tokens.

### Web document assessments

| Field | Meaning |
|---|---|
| `dataset_revision` | Upstream dataset revision from which the document was collected |
| `upstream_relevant` | Original upstream relevance label, where available; retained RedSage rows have the string `true` |
| `domain_score` | Classifier score for cybersecurity or supporting computing relevance |
| `substance_score` | Classifier score for technically substantive content |
| `usability_score` | Diagnostic text-usability score; not a selection criterion |
| `joint_score` | Minimum of the domain and substance scores |
| `model_sha256` | SHA-256 identifier of the original classifier metadata, also recorded in `selection-policy.json` |

The approved semantic selection rule required `domain_score >= 0.80` and
`substance_score >= 0.80`. RedSage also required an upstream relevance label of
`true`. Later deduplication and corpus curation determine which qualifying
documents appear in the retained corpus. These scores approximate Qwen labels;
they are not probabilities of factual correctness. The classifier identifier
and weight checksum identify the fitted model; its weights are not included in
this metadata supplement.

### Transcript document assessments

`route` records the original eligibility result (`eligible` for these retained
documents). `scoring_complete` records whether passage scoring completed.

### Transcript passage assessments

| Fields | Meaning |
|---|---|
| `record_id`, `content_hash` | Link to the exact full transcript |
| `start`, `end` | Zero-based Unicode character positions: `content[start:end]`, end exclusive |
| `model`, `model_revision` | Qwen model name and immutable checkpoint revision |
| `prompt_version` | Identifier of the classification rubric |
| `parse_status` | Whether the model response satisfied the expected schema |
| `domain`, `quality`, `language` | Original parsed classifications |
| `reason` | Original short model-generated explanation |

For retained transcripts, the scored passages cover the full text without gaps
or overlaps. Each passage has `parse_status = ok`, domain `security` or
`computing`, quality `useful`, and language `english`. The exact classification
rubric, inference settings and selection rules are recorded in
`selection-policy.json`.

## Loading

Authenticate with a Hugging Face account that has access to CyberOpen. For
example, download a metadata table without changing the training-data loader:

```python
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

path = hf_hub_download(
    repo_id="cyberopen-research/CyberOpen",
    filename="metadata/filtering/web-records.parquet",
    repo_type="dataset",
    token=True,
)
assessments = pq.read_table(path)
```

Use the same repository revision when loading training data and metadata for a
reproducible experiment. The repository's root `SHA256SUMS` covers all files in
this supplement; `selection-policy.json` additionally records table checksums.

## Coverage and interpretation

These files describe retained documents, not the complete accepted/rejected
candidate population. They do not contain duplicate-removal lineage or
independent human-review labels. Native empty `selection_decision` columns are
omitted; no synthetic keep/drop labels have been introduced. Changing a cutoff
within this supplement can select a subset of retained documents but cannot
recover candidates excluded before release.

Document-level Q&A assessments already present in the training schema remain
there. This supplement adds the separate web and transcript assessments; it
does not replace the other source-specific metadata.

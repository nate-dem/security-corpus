"""Export retained-record filtering metadata without execution or local metadata.

This is a column projection of historical assessments, not a new selection or
model run. Input checksums bind the export to the saved filtering tables.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.curation_single import rubric


COLUMNS = {
    "web-records.parquet": (
        "record_id", "content_hash", "source", "content_length",
        "dataset_revision", "upstream_relevant", "domain_score",
        "substance_score", "usability_score", "joint_score", "model_sha256",
    ),
    "youtube-records.parquet": (
        "record_id", "content_hash", "source", "content_length", "route",
        "scoring_complete",
    ),
    "youtube-spans.parquet": (
        "record_id", "content_hash", "start", "end", "model", "model_revision",
        "prompt_version", "parse_status", "domain", "quality", "language", "reason",
    ),
}
# Deployment identifiers must never enter the published projection. Generic
# discussion of hostnames or Slurm in a source-text assessment is legitimate.
PRIVATE_TEXT = re.compile(
    r"(?i)(/Users/|/home/|/scratch/|/Volumes/|/private/|/tmp/|"
    r"[a-z]:\\Users\\|"
    r"\b(?:slurm_job_id|hostname|job_id)\s*[=:])"
)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def project_table(source, destination, columns):
    """Preserve values/types/order, strip every unlisted column and metadata key."""
    reader = pq.ParquetFile(source)
    schema = pa.schema([
        pa.field(name, reader.schema_arrow.field(name).type,
                 nullable=reader.schema_arrow.field(name).nullable)
        for name in columns
    ])
    count = 0
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(destination, schema, compression="zstd") as writer:
        for batch in reader.iter_batches(batch_size=8192, columns=list(columns)):
            for name in columns:
                array = batch.column(batch.schema.get_field_index(name))
                if pa.types.is_string(array.type) or pa.types.is_large_string(array.type):
                    if any(PRIVATE_TEXT.search(value) for value in array.to_pylist()
                           if value is not None):
                        raise ValueError(f"Private metadata found in {source.name}:{name}")
            clean = pa.RecordBatch.from_arrays(batch.columns, schema=schema)
            writer.write_batch(clean)
            count += len(clean)
    if count != reader.metadata.num_rows:
        raise ValueError("Row count changed during projection")
    return {"rows": count, "columns": list(columns), "sha256": sha(destination)}


def transcript_settings(source, configs):
    hashes = pq.read_table(source / "youtube-spans.parquet", columns=["run_config_sha256"])
    needed = set(hashes.column(0).unique().to_pylist())
    found, settings = set(), set()
    fields = ("model", "model_revision", "prompt_version", "temperature",
              "enable_thinking", "focal_tokens", "max_model_len",
              "max_output_tokens", "tokenizer_files", "versions")
    for path in Path(configs).glob("*/work/score/run-config.json"):
        digest = sha(path)
        if digest not in needed:
            continue
        config = json.loads(path.read_text())
        if config["code"]["scripts.curation_single.rubric"] != sha(Path(rubric.__file__)):
            raise ValueError("Recorded transcript rubric differs from exported rubric")
        found.add(digest)
        settings.add(json.dumps({key: config[key] for key in fields}, sort_keys=True))
    if found != needed or len(settings) != 1:
        raise ValueError("Transcript inference configuration is missing or inconsistent")
    return json.loads(settings.pop())


def prepare(source, output, model_path, dataset_revision, youtube_configs):
    source, output, model_path = map(Path, (source, output, model_path))
    if output.exists():
        raise ValueError("Choose a new output folder")
    inventory = json.loads((source / "README.json").read_text())
    model = json.loads(model_path.read_text())
    model_hash = sha(model_path)
    for filename in COLUMNS:
        if sha(source / filename) != inventory["tables"][filename]["sha256"]:
            raise ValueError(f"Input checksum failed: {filename}")
    web_models = pq.read_table(source / "web-records.parquet", columns=["model_sha256"])
    if web_models.column(0).unique().to_pylist() != [model_hash]:
        raise ValueError("Classifier metadata does not match the scored records")
    versions = pq.read_table(source / "youtube-spans.parquet", columns=["prompt_version"])
    if versions.column(0).unique().to_pylist() != [rubric.VERSION]:
        raise ValueError("Transcript rubric does not match recorded prompt version")
    inference = transcript_settings(source, youtube_configs)
    tables = {name: project_table(source / name, output / name, fields)
              for name, fields in COLUMNS.items()}
    for name, item in tables.items():
        if item["rows"] != inventory["tables"][name]["rows"]:
            raise ValueError("Input coverage changed")
    policy = {
        "dataset": "cyberopen-research/CyberOpen",
        "training_data_revision": dataset_revision,
        "scope": "Filtering assessments for retained Primus-FineWeb, RedSage-CFW and YouTube documents.",
        "join_keys": ["record_id", "content_hash"],
        "reference_tokenizer": "cl100k_base",
        "tables": tables,
        "web": {
            "selection": {"domain_score_gte": 0.80, "substance_score_gte": 0.80,
                          "redsage_upstream_relevant": "true"},
            "joint_score": "min(domain_score, substance_score)",
            "usability_score": "Diagnostic only; not used for selection.",
            "classifier": {"type": "logistic_regression", "metadata_sha256": model_hash,
                           **{key: model[key] for key in (
                               "version", "weights_sha256", "features", "targets",
                               "sklearn_version", "documents", "spans")}},
        },
        "youtube": {
            "selection": {"parse_status": "ok", "domain": ["security", "computing"],
                          "quality": "useful", "language": "english",
                          "all_passages_required": True, "scoring_complete": True},
            "offsets": "Zero-based Unicode character indices into content; start inclusive, end exclusive.",
            "prompt_version": rubric.VERSION,
            "inference": inference,
            "system_prompt": rubric.SYSTEM,
            "label_values": rubric.LABELS,
        },
    }
    text = json.dumps(policy, indent=2, ensure_ascii=False) + "\n"
    if PRIVATE_TEXT.search(text):
        raise ValueError("Private metadata in selection policy")
    (output / "selection-policy.json").write_text(text)
    (output / "README.md").write_text(README)
    return tables


README = """# CyberOpen filtering metadata

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
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--web-model", type=Path, required=True)
    parser.add_argument("--youtube-configs", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.web_model,
                             args.dataset_revision, args.youtube_configs), indent=2))


if __name__ == "__main__":
    main()

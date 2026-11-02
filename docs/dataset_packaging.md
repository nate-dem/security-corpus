# Dataset packaging and loading

The dataset repository contains the training Parquet files, dataset card, file
configuration, filtering metadata, and download checksums. Corpus archives, local
build manifests,
audit databases, excluded examples, and development notes are not uploaded.
The technical report is maintained separately in this code repository.

## Prepare a dataset directory

Use a separate environment for Hub packaging:

```bash
python3 -m venv .venv-hf
.venv-hf/bin/python -m pip install -r scripts/release/requirements-huggingface.txt
```

Starting with a completed, audited corpus directory:

```bash
.venv-hf/bin/python scripts/release/prepare_huggingface.py \
  --source /path/to/completed-corpus \
  --output data/huggingface/CyberOpen \
  --local-manifest reports/release/cyberopen/package-manifest.json
```

The destination must be new. The packager verifies the source bindings and every
training-file checksum, checks source totals, and copies the Parquet bytes
without rewriting the records. Numbered subdirectories bound the number of
entries per folder. The README declares the full union schema and selects only
`data/*/*.parquet` as the `train` split.

The local manifest records the source bindings and file mapping outside the
upload folder. It is used to verify the prepared package:

```bash
.venv-hf/bin/python scripts/release/verify_huggingface.py \
  data/huggingface/CyberOpen \
  --local-manifest reports/release/cyberopen/package-manifest.json \
  --report reports/release/cyberopen/verification.json
```

Verification streams every record through Hugging Face Datasets and compares
source document/token totals with the completed corpus.

## Prepare filtering metadata

The [filtering metadata guide](filtering_metadata.md) describes the retained web
and transcript assessments. Export the research fields from the original bound
filtering tables using:

```bash
.venv-hf/bin/python -m scripts.release.prepare_filtering_supplement \
  --source /path/to/filtering-tables \
  --web-model /path/to/web-classifier/model.json \
  --youtube-configs /path/to/transcript-scoring/parts \
  --dataset-revision 8da37a61b56b4228fc299ff795299288cf5a4c69 \
  --output data/huggingface/CyberOpen/metadata/filtering
```

The destination must be new. The exporter verifies input checksums and recorded
rubric/configuration identities, then preserves the original values and row
order for an explicit set of research fields. It removes execution metadata and
embedded Arrow metadata. It does not rerun filtering or change the training
documents. The exported `selection-policy.json` records the rules, model
identifiers, transcript rubric and inference settings.

After adding this supplement, regenerate the package checksums before uploading:

```python
import hashlib
from pathlib import Path

package = Path("data/huggingface/CyberOpen")
lines = []
for path in sorted(package.rglob("*")):
    relative = path.relative_to(package)
    if (not path.is_file() or str(relative) == "SHA256SUMS"
            or relative.parts[0] == ".cache" or path.name == ".DS_Store"):
        continue
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    lines.append(f"{digest}  {relative}\n")
(package / "SHA256SUMS").write_text("".join(lines))
```

The uploader accepts only the named final supplement files and the training
package files. The default dataset configuration continues to select only
`data/*/*.parquet`; metadata tables are downloaded separately.

## Upload privately

Authenticate using an account with write access to the destination organization:

```bash
.venv-hf/bin/hf auth login
.venv-hf/bin/python scripts/release/upload_huggingface_private.py \
  'YOUR_ORG/CyberOpen'
```

The uploader checks the package before creating a private dataset repository. It
refuses an existing public destination or one containing files outside this
release. Rerunning the command can resume an interrupted upload. Organization
members need read access to view and download a private dataset.

## Load training documents

```python
from datasets import load_dataset

corpus = load_dataset('cyberopen-research/CyberOpen', split='train', streaming=True)
for document in corpus:
    text = document['content']
    source = document['source_id']
```

Authenticate before accessing a private repository. Pin a Hub commit with
`revision` when reproducibility requires a fixed dataset version.

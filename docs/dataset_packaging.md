# Dataset packaging and loading

The dataset repository contains the training Parquet files, dataset card, file
configuration, and download checksums. Corpus archives, local build manifests,
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

corpus = load_dataset('YOUR_ORG/CyberOpen', split='train', streaming=True)
for document in corpus:
    text = document['content']
    source = document['source_id']
```

Authenticate before accessing a private repository. Pin a Hub commit with
`revision` when reproducibility requires a fixed dataset version.

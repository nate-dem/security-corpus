"""Open a finalized corpus using an explicit union of source-specific fields."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds

from scripts.release.finalize_local import contained, sha


def load_dataset(root):
    root = Path(root).resolve()
    prepared = json.loads((root / "final-prepared.json").read_text())
    if sha(root / "schema.json") != prepared["schema_sha256"]:
        raise ValueError("Schema description changed")
    if sha(root / "corpus-manifest.json") != prepared["manifest_sha256"]:
        raise ValueError("Corpus manifest changed")
    descriptor = json.loads((root / "schema.json").read_text())
    relative = descriptor["union_schema"]
    path = contained(root, relative)
    if sha(path) != prepared["evidence_sha256"][relative]:
        raise ValueError("Union schema changed")
    schema = pa.ipc.read_schema(pa.BufferReader(path.read_bytes()))
    manifest = json.loads((root / "corpus-manifest.json").read_text())
    return ds.dataset([str(contained(root, e["relative_path"])) for e in manifest["files"]],
                      schema=schema, format="parquet")

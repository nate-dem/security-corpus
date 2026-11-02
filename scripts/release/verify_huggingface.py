"""Verify package checksums and stream every record through Hugging Face Datasets."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import tempfile
import sys

import pyarrow as pa
from datasets import load_dataset, load_dataset_builder

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.release.upload_huggingface_private import verify_package, sha


def verify(root, report, local_manifest):
    root = Path(root).resolve()
    inventory = verify_package(root)
    manifest = json.loads(Path(local_manifest).read_text())
    for entry in manifest["training_files"]:
        if inventory["files"].get(entry["package_path"]) != entry["sha256"]:
            raise ValueError("Training files differ from the local source binding")
    expected = {str(root / entry["package_path"]) for entry in manifest["training_files"]}
    union = pa.ipc.read_schema(pa.BufferReader(Path(manifest["schema_source"]).read_bytes()))
    counts = defaultdict(lambda: dict(records=0, tokens=0))
    with tempfile.TemporaryDirectory(prefix="hf-corpus-check-") as cache:
        builder = load_dataset_builder(str(root), cache_dir=cache)
        assert set(builder.config.data_files) == {"train"}
        assert set(builder.config.data_files["train"]) == expected
        assert builder.info.features.arrow_schema.equals(union, check_metadata=False)
        print(f"Configuration selects {len(expected):,} training files and {len(union)} columns.", flush=True)
        data = load_dataset(str(root), split="train", streaming=True, cache_dir=cache)
        for index, table in enumerate(data.with_format("arrow").iter(batch_size=1024)):
            assert table.schema.equals(union, check_metadata=False)
            assert table["content_length"].null_count == 0
            for row in table.group_by("source_id").aggregate(
                [("content_length", "sum"), ("content_length", "count")]).to_pylist():
                counts[row["source_id"]]["records"] += row["content_length_count"]
                counts[row["source_id"]]["tokens"] += row["content_length_sum"]
            if (index + 1) % 500 == 0:
                print(f"Streamed {(index + 1) * 1024:,} records", flush=True)
    if dict(counts) != manifest["by_source"]:
        raise ValueError("HF-loaded source counts differ from the snapshot")
    files = [p for p in root.rglob("*") if p.is_file()]
    result = dict(
        verified_at=datetime.now(timezone.utc).isoformat(), package=str(root),
        snapshot=manifest["snapshot"], snapshot_manifest_sha256=manifest["snapshot_manifest_sha256"],
        local_manifest_sha256=sha(Path(local_manifest)),
        sha256sums_sha256=sha(root / "SHA256SUMS"),
        package_files=len(files), package_bytes=sum(p.stat().st_size for p in files),
        training_files=len(expected), training_bytes=manifest["totals"]["bytes"],
        documents=sum(v["records"] for v in counts.values()),
        tokens=sum(v["tokens"] for v in counts.values()), sources=len(counts),
        schema_fields=len(union), split_names=["train"], all_checksums_passed=True,
        all_training_files_streamed=True, evidence_excluded_from_training=True,
        all_source_columns_retained=True, payload_bytes_unchanged=True,
        remote_upload_performed=False,
        versions={k: importlib.metadata.version(k) for k in ["datasets", "huggingface_hub", "pyarrow"]},
    )
    report = Path(report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--local-manifest", type=Path, required=True)
    args = parser.parse_args()
    verify(args.folder, args.report, args.local_manifest)

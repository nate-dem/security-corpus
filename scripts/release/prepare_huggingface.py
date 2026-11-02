"""Package a completed snapshot for HF without rewriting its Parquet files.

Requires datasets and PyYAML in addition to PyArrow. No network calls are made.
The existing union schema is presentation metadata, not a new corpus schema.
"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from datasets import Features


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def contained(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe relative path: {relative}")
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes package: {relative}")
    return path


def copy_independent(src, dst):
    """Use APFS copy-on-write when available; never hardlink or symlink."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        clone = ctypes.CDLL(None, use_errno=True).clonefile
        clone.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        clone.restype = ctypes.c_int
        if clone(os.fsencode(src), os.fsencode(dst), 0) == 0:
            return
    shutil.copy2(src, dst)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def package(source, output, template, local_manifest, technical_report=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    local_manifest = Path(local_manifest).resolve()
    if local_manifest.is_relative_to(output):
        raise ValueError("The local build manifest must be outside the upload folder")
    if technical_report is not None:
        technical_report = Path(technical_report).resolve()
        if not technical_report.is_file():
            raise ValueError("Technical report PDF does not exist")
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Choose a new output directory outside the source snapshot")
    complete = json.loads((source / "final-corpus-complete.json").read_text())
    if not complete.get("complete") or not complete.get("integrity_passed"):
        raise ValueError("Snapshot is not complete with a passing integrity audit")
    for relative, key in [("corpus-manifest.json", "manifest_sha256"),
                          ("final-prepared.json", "prepared_sha256"),
                          ("quality/summary.json", "quality_summary_sha256")]:
        if sha(source / relative) != complete[key]:
            raise ValueError(f"Snapshot binding failed: {relative}")
    manifest = json.loads((source / "corpus-manifest.json").read_text())
    prepared = json.loads((source / "final-prepared.json").read_text())
    summary = json.loads((source / "quality/summary.json").read_text())
    if sha(source / "schema.json") != prepared["schema_sha256"]:
        raise ValueError("Snapshot schema binding failed")
    descriptor = json.loads((source / "schema.json").read_text())
    union_path = descriptor["union_schema"]
    if sha(contained(source, union_path)) != prepared["evidence_sha256"][union_path]:
        raise ValueError("Union schema binding failed")
    schema = pa.ipc.read_schema(pa.BufferReader((source / union_path).read_bytes()))
    features = Features.from_arrow_schema(schema)
    original_paths = [entry["relative_path"] for entry in manifest["files"]]
    if len(set(original_paths)) != len(original_paths):
        raise ValueError("Duplicate training paths in manifest")
    actual_records = {str(p.relative_to(source)) for p in (source / "records").rglob("*") if p.is_file()}
    if actual_records != set(original_paths):
        raise ValueError("Training file inventory does not match manifest")
    output.mkdir(parents=True)
    mapping, sources = [], defaultdict(lambda: {"records": 0, "tokens": 0})
    for index, entry in enumerate(manifest["files"]):
        old = contained(source, entry["relative_path"])
        relative = f"data/{index // 1000:03d}/{old.name}"
        new = contained(output, relative)
        copy_independent(old, new)
        if new.stat().st_size != entry["bytes"] or sha(new) != entry["sha256"]:
            raise ValueError(f"Training checksum mismatch: {old}")
        parquet = pq.ParquetFile(new)
        # No field may be silently lost by the HF features declaration.
        for field in parquet.schema_arrow:
            if field.name not in schema.names or field.type != schema.field(field.name).type:
                raise ValueError(f"Field outside union schema: {old}: {field}")
        count, tokens = 0, 0
        for batch in parquet.iter_batches(columns=["source_id", "content_length"]):
            grouped = pa.Table.from_batches([batch]).group_by("source_id").aggregate(
                [("content_length", "sum"), ("content_length", "count")])
            for row in grouped.to_pylist():
                if row["source_id"] is None:
                    raise ValueError(f"Missing source: {old}")
                n, t = row["content_length_count"], row["content_length_sum"]
                sources[row["source_id"]]["records"] += n
                sources[row["source_id"]]["tokens"] += t
                count += n
                tokens += t
        if count != parquet.metadata.num_rows or count != entry["records"] or tokens != entry["tokens"]:
            raise ValueError(f"Training counts mismatch: {old}")
        mapping.append(dict(original_path=entry["relative_path"], package_path=relative,
                            sha256=entry["sha256"], bytes=entry["bytes"], records=count, tokens=tokens))
        if (index + 1) % 1000 == 0:
            print(f"Verified {index + 1:,}/{len(original_paths):,} training files", flush=True)
    if dict(sources) != summary["by_source"]:
        raise ValueError("Source counts differ from completed audit")
    if technical_report is not None:
        copy_independent(technical_report, output / "technical_report.pdf")
        if sha(technical_report) != sha(output / "technical_report.pdf"):
            raise ValueError("Technical report copy mismatch")
    totals = {"records": sum(e["records"] for e in mapping),
              "tokens": sum(e["tokens"] for e in mapping),
              "bytes": sum(e["bytes"] for e in mapping)}
    if totals != complete["totals"]:
        raise ValueError("Completed snapshot totals differ")
    metadata = {
        "pretty_name": "CyberOpen",
        "task_categories": ["text-generation"],
        "tags": ["cybersecurity", "continued-pretraining", "research-corpus"],
        "size_categories": ["1M<n<10M"],
        "configs": [{"config_name": "default", "default": True,
                     "data_files": [{"split": "train", "path": "data/*/*.parquet"}]}],
        "dataset_info": {"features": features._to_yaml_list()},
    }
    replacements = {
        "{{DOCUMENTS}}": f"{totals['records']:,}",
        "{{TOKENS}}": f"{totals['tokens']:,}",
        "{{TRAINING_FILES}}": f"{len(mapping):,}",
        "{{SOURCE_TABLE}}": "\n".join(f"| `{s}` | {v['records']:,} | {v['tokens']:,} |" for s, v in sorted(sources.items())),
        "{{SCHEMA_FIELDS}}": str(len(schema)),
        "{{REPORT_REFERENCE}}": (
            "See the [technical report](technical_report.pdf) for source selection, "
            "filtering, deduplication, and dataset analysis."
            if technical_report else
            "The accompanying technical report describes source selection, filtering, "
            "deduplication, and dataset analysis."
        ),
    }
    body = Path(template).read_text()
    for key, value in replacements.items():
        body = body.replace(key, value)
    if "{{" in body:
        raise ValueError("Unfilled dataset-card template")
    (output / "README.md").write_text("---\n" + yaml.safe_dump(metadata, sort_keys=False) + "---\n\n" + body)
    (output / ".gitattributes").write_text(
        "*.parquet filter=lfs diff=lfs merge=lfs -text\n"
        "*.pdf filter=lfs diff=lfs merge=lfs -text\n")
    local_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_json(local_manifest, dict(
        dataset_name="CyberOpen", source_root=str(source), package_root=str(output),
        snapshot=source.name, snapshot_manifest_sha256=complete["manifest_sha256"],
        payload_bytes_unchanged=True, row_order_unchanged=True,
        schema_source=str(source / union_path),
        totals=totals, by_source=dict(sources), training_files=mapping,
        technical_report_source=str(technical_report) if technical_report else None))
    for folder in [output, *[p for p in output.rglob("*") if p.is_dir()]]:
        if len(list(folder.iterdir())) >= 10000:
            raise ValueError(f"HF directory entry limit: {folder}")
    checksums = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            checksums.append(f"{sha(path)}  {path.relative_to(output)}\n")
    (output / "SHA256SUMS").write_text("".join(checksums))
    print(json.dumps(dict(output=str(output), totals=totals,
                         training_files=len(mapping), local_manifest=str(local_manifest)), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-manifest", type=Path, required=True)
    parser.add_argument("--technical-report", type=Path)
    parser.add_argument("--template", type=Path,
                        default=Path(__file__).resolve().parents[2] / "docs/huggingface_dataset_card.md.in")
    args = parser.parse_args()
    package(args.source, args.output, args.template, args.local_manifest, args.technical_report)


if __name__ == "__main__":
    main()

"""Normalize existing physical field types without changing corpus values.

Writes a new candidate. Null-only integer columns may become their documented
string, timestamp, boolean or numeric types; non-null integers in those columns
are rejected. Source-specific columns remain present only where they existed.
An explicit union schema supports streaming across all sources.
"""

import argparse
import json
import logging
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from scripts.release import finalize_local as final
from scripts.release.correct_licenses import write
from scripts.release.export_canonical import BASE_FIELDS

LOG = logging.getLogger(__name__)
REVISION = "physical-schema-20261024"
TYPES = {
    **{name: pa.string() for name in BASE_FIELDS},
    "content": pa.large_string(),
    "content_length": pa.int64(),
    "ingested_at": pa.timestamp("us", tz="UTC"),
    "published_at": pa.timestamp("us", tz="UTC"),
    "has_accepted_answer": pa.bool_(),
    "closed": pa.bool_(),
    "severity": pa.string(),
    "cvss_score": pa.float64(),
    "exploited_in_wild": pa.bool_(),
}


def normalize(table):
    """Only lossless string widening and typing of absent values are allowed."""
    after = table
    for name, dtype in TYPES.items():
        if name not in table.column_names:
            final.require(name not in BASE_FIELDS, f"Missing canonical field: {name}")
            continue
        col = table[name]
        if col.type == dtype:
            continue
        if col.null_count == len(col):
            value = pa.nulls(len(col), type=dtype)
        else:
            final.require(
                name == "content" and col.type == pa.string(),
                f"Non-null field cannot be reinterpreted: {name}: {col.type}",
            )
            value = col.cast(dtype, safe=True)
            final.require(value.cast(col.type).equals(col), "String values changed")
        field = table.schema.field(name).with_type(dtype)
        after = after.set_column(after.schema.get_field_index(name), field, value)
    return after


def verify_parent(parent):
    prepared, manifest = final.verify_prepared(parent)
    marker = final.read(parent / "final-corpus-complete.json")
    final.require(marker["complete"] and marker["integrity_passed"], "Parent incomplete")
    for key, name in (
        ("manifest_sha256", "corpus-manifest.json"),
        ("prepared_sha256", "final-prepared.json"),
        ("quality_summary_sha256", "quality/summary.json"),
    ):
        final.require(marker[key] == final.sha(parent / name), "Parent binding changed")
    return prepared, manifest


def build(parent, output):
    parent, output = parent.resolve(), output.resolve()
    final.require(not output.exists(), "Use a new output directory")
    final.require(not output.is_relative_to(parent), "Output overlaps parent")
    prepared, manifest = verify_parent(parent)
    output.mkdir(parents=True)
    (output / "records").mkdir()
    shutil.copytree(parent / "evidence", output / "evidence", copy_function=final.link_or_copy)
    evidence = output / "evidence" / REVISION
    evidence.mkdir()
    for name in ("schema.json", "final-prepared.json", "corpus-manifest.json", "final-corpus-complete.json"):
        shutil.copyfile(parent / name, evidence / ("parent-" + name))
    shutil.copyfile(Path(__file__), evidence / "applied-code.py")
    final.link_or_copy(parent / "record-metadata.jsonl.gz", output / "record-metadata.jsonl.gz")
    files, changes, schemas = [], [], []
    for ordinal, entry in enumerate(manifest["files"]):
        source = final.contained(parent, entry["relative_path"])
        target = final.contained(output, entry["relative_path"])
        final.require(final.sha(source) == entry["sha256"], "Parent payload differs")
        with pq.ParquetFile(source) as pf:
            altered = {f.name: dict(before=str(f.type), after=str(TYPES[f.name]))
                       for f in pf.schema_arrow if f.name in TYPES and f.type != TYPES[f.name]}
            if altered:
                writer = None
                try:
                    for batch in pf.iter_batches(batch_size=128):
                        table = normalize(pa.Table.from_batches([batch]))
                        if writer is None:
                            writer = pq.ParquetWriter(target, table.schema, compression="zstd")
                        writer.write_table(table)
                    if writer is None:
                        table = normalize(pf.read())
                        pq.write_table(table, target, compression="zstd")
                finally:
                    if writer is not None:
                        writer.close()
                with pq.ParquetFile(target) as checked:
                    groups = 0
                    for group, batch in enumerate(pf.iter_batches(batch_size=128)):
                        final.require(
                            normalize(pa.Table.from_batches([batch])).equals(
                                checked.read_row_group(group), check_metadata=True
                            ), "Parquet roundtrip changed retained values",
                        )
                        groups += 1
                    final.require(groups == checked.num_row_groups or not checked.metadata.num_rows,
                                  "Output row groups differ")
                changes.append(dict(file=ordinal, columns=altered))
            else:
                final.link_or_copy(source, target)
        final.require(final.sha(source) == entry["sha256"], "Parent changed during rewrite")
        schemas.append(pq.read_schema(target))
        files.append(dict(entry, path=str(target), sha256=final.sha(target),
                          bytes=target.stat().st_size, parent_sha256=entry["sha256"]))
        if ordinal % 250 == 0 or ordinal + 1 == len(manifest["files"]):
            LOG.info("Schema rewrite %s/%s", ordinal + 1, len(manifest["files"]))
    common = [pa.schema([s.field(n) for n in BASE_FIELDS]) for s in schemas]
    final.require(all(s.equals(common[0]) for s in common), "Canonical schemas differ")
    union = pa.unify_schemas(schemas)
    (evidence / "union-schema.arrow").write_bytes(union.serialize().to_pybytes())
    dataset = ds.dataset([e["path"] for e in files], schema=union, format="parquet")
    streamed_rows = sum(batch.num_rows for batch in dataset.scanner(
        batch_size=128, batch_readahead=1, fragment_readahead=1
    ).to_batches())
    final.require(streamed_rows == manifest["totals"]["records"], "Cross-shard scan count differs")
    totals = {key: sum(e[key] for e in files) for key in ("records", "tokens", "bytes")}
    for key in ("records", "tokens"):
        final.require(totals[key] == manifest["totals"][key], "Corpus selection changed")
    report = dict(complete=True, revision=REVISION, parent_manifest_sha256=prepared["manifest_sha256"],
                  changed_files=len(changes), changes=changes, totals=totals,
                  all_values_and_nulls_preserved=True, canonical_schema_identical=True,
                  strict_full_schema_union_passed=True, full_union_scan_rows=streamed_rows,
                  release_ready=False)
    write(evidence / "revision.json", report)
    schema = final.read(parent / "schema.json")
    schema.update(physical_types={name: str(dtype) for name, dtype in TYPES.items()},
                  union_schema=f"evidence/{REVISION}/union-schema.arrow",
                  loading="Use load_final_dataset.py to preserve all source-specific columns across shards.")
    write(output / "schema.json", schema)
    write(output / "corpus-manifest.json", dict(manifest, files=files, totals=totals,
                                               parent_manifest_sha256=prepared["manifest_sha256"]))
    write(output / "final-prepared.json", dict(
        prepared, manifest_sha256=final.sha(output / "corpus-manifest.json"),
        record_metadata_sha256=final.sha(output / "record-metadata.jsonl.gz"),
        schema_sha256=final.sha(output / "schema.json"), code_sha256=final.sha(Path(__file__)),
        totals=totals, evidence_sha256={str(p.relative_to(output)): final.sha(p)
                                      for p in (output / "evidence").rglob("*") if p.is_file()},
    ))
    final.require(verify_parent(parent)[0] == prepared, "Parent evidence changed")
    return {key: value for key, value in report.items() if key != "changes"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(build(args.parent, args.output), indent=2))

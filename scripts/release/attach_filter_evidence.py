"""Package verified filtering sidecars with a new, text-identical corpus revision.

The scores remain in their original tabular schemas. No canonical record fields,
licenses, selection decisions or text change. All evidence is bound by checksum
and the new candidate receives its own integrity audit.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sqlite3

import pyarrow.parquet as pq

from scripts.release import finalize_local as final
from scripts.release.correct_licenses import write
from scripts.release.normalize_final_schema import verify_parent
from scripts.release.restore_filter_evidence import SOURCES

REVISION = "filtering-restoration-20261024"


def verify_bundle(parent, manifest, prepared, bundle):
    report = final.read(bundle / "summary.json")
    final.require(report["complete"] and report["manifest_sha256"] == prepared["manifest_sha256"],
                  "Filtering evidence incomplete or bound to another candidate")
    quality = final.read(parent / "quality/summary.json")
    expected = {s: quality["by_source"][s]["records"] for s in SOURCES}
    final.require(report["retained_by_source"] == expected
                  and report["record_coverage"] == sum(expected.values()), "Evidence coverage differs")
    final.require(set(report["files"]) == {"web-records.parquet", "youtube-records.parquet", "youtube-spans.parquet"},
                  "Unexpected evidence tables")
    for name, digest in report["files"].items():
        final.require(final.sha(bundle / name) == digest, "Evidence table changed")
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE rows (file INTEGER, row INTEGER, id TEXT UNIQUE, hash TEXT, tokens INTEGER, source TEXT, PRIMARY KEY(file,row))")
        counts = Counter()
        columns = ["final_file", "final_row", "record_id", "content_hash", "content_length", "source"]
        for name in ("web-records", "youtube-records"):
            n = 0
            with pq.ParquetFile(bundle / (name+".parquet")) as pf:
                for batch in pf.iter_batches(batch_size=4096, columns=columns):
                    rows = batch.to_pylist()
                    for row in rows:
                        final.require(row["source"] in SOURCES, "Unexpected scored source")
                        final.require((row["source"] == "youtube-commons") == (name == "youtube-records"),
                                      "Record is in the wrong evidence table")
                    db.executemany("INSERT INTO rows VALUES (?,?,?,?,?,?)", [tuple(r[c] for c in columns) for r in rows])
                    counts.update(r["source"] for r in rows)
                    n += len(rows)
            final.require(n == report["evidence_rows"][name], "Evidence row count differs")
        final.require(dict(counts) == expected, "Actual evidence coverage differs")
        checked = 0
        for ordinal, entry in enumerate(manifest["files"]):
            requests = {r[0]:r[1:] for r in db.execute("SELECT row,id,hash,tokens,source FROM rows WHERE file=?", (ordinal,))}
            if not requests:
                continue
            path = final.contained(parent, entry["relative_path"])
            final.require(final.sha(path) == entry["sha256"], "Parent payload changed")
            with pq.ParquetFile(path) as pf:
                offset = 0
                for batch in pf.iter_batches(batch_size=2048, columns=["record_id", "content_hash", "content_length", "source_id"]):
                    for i, row in enumerate(batch.to_pylist()):
                        proof = requests.pop(offset+i, None)
                        if proof is not None:
                            final.require(tuple(row[c] for c in ("record_id","content_hash","content_length","source_id")) == proof,
                                          "Evidence record address/identity differs")
                            checked += 1
                    offset += len(batch)
            final.require(not requests, "Evidence row outside payload")
        final.require(checked == sum(expected.values()), "Unverified evidence records")
        # Every span must refer to its retained YouTube document and row address.
        spans = 0
        with pq.ParquetFile(bundle / "youtube-spans.parquet") as pf:
            for batch in pf.iter_batches(batch_size=2048, columns=["record_id","content_hash","final_file","final_row"]):
                for row in batch.to_pylist():
                    proof = db.execute("SELECT hash,file,row,source FROM rows WHERE id=?", (row["record_id"],)).fetchone()
                    final.require(proof == (row["content_hash"],row["final_file"],row["final_row"],"youtube-commons"),
                                  "Span has no matching retained document")
                    spans += 1
        final.require(spans == report["evidence_rows"]["youtube-spans"], "Span row count differs")
    return report


def build(parent, bundle, output):
    parent, bundle, output = [p.resolve() for p in (parent,bundle,output)]
    final.require(not output.exists() and not output.is_relative_to(parent), "Use a new output directory")
    prepared, manifest = verify_parent(parent)
    report = verify_bundle(parent, manifest, prepared, bundle)
    binding = final.read(parent / "quality/binding.json")
    final.require(binding["manifest_sha256"] == prepared["manifest_sha256"]
                  and binding["code_sha256"] == final.sha(Path(final.__file__)), "Parent audit cache differs")
    output.mkdir()
    (output / "records").mkdir()
    shutil.copytree(parent / "evidence", output / "evidence", copy_function=final.link_or_copy)
    evidence = output / "evidence" / REVISION
    evidence.mkdir()
    for name in (*report["files"], "summary.json", "input-plan.json"):
        final.link_or_copy(bundle / name, evidence / name)
    for name in ("restore_filter_evidence.py", "attach_filter_evidence.py"):
        shutil.copyfile(Path(__file__).with_name(name), evidence / name)
    for name in ("corpus-manifest.json", "final-prepared.json", "final-corpus-complete.json"):
        shutil.copyfile(parent / name, evidence / ("parent-"+name))
    files = []
    for entry in manifest["files"]:
        source = final.contained(parent, entry["relative_path"])
        final.require(final.sha(source) == entry["sha256"], "Parent payload differs")
        target = final.contained(output, entry["relative_path"])
        final.link_or_copy(source, target)
        files.append(dict(entry, path=str(target)))
    for name in ("record-metadata.jsonl.gz", "schema.json"):
        final.link_or_copy(parent / name, output / name)
    write(output / "corpus-manifest.json", dict(manifest, files=files, parent_manifest_sha256=prepared["manifest_sha256"]))
    descriptor = dict(complete=True, corpus_manifest_sha256=final.sha(output / "corpus-manifest.json"),
        restored_against_parent_manifest_sha256=prepared["manifest_sha256"],
        payloads_byte_identical_to_parent=True, record_metadata_unchanged=True,
        record_coverage=report["record_coverage"], retained_by_source=report["retained_by_source"],
        tables={name:dict(relative_path=str((evidence/name).relative_to(output)), sha256=digest,
                         rows=report["evidence_rows"][name.removesuffix(".parquet")])
                for name,digest in report["files"].items()},
        join_key="record_id; content_hash verifies the text; final_file/final_row are zero-based manifest addresses",
        scope="Original filtering evidence; not new semantic evaluation. YouTube spans are one-to-many. Native null selection_decision values are preserved.")
    write(output / "filtering-evidence.json", descriptor)
    write(evidence / "README.json", descriptor)
    write(output / "final-prepared.json", dict(prepared,
        manifest_sha256=final.sha(output / "corpus-manifest.json"), code_sha256=final.sha(Path(__file__)),
        filtering_evidence_restored=True, filtering_evidence_descriptor="filtering-evidence.json",
        evidence_sha256={str(p.relative_to(output)):final.sha(p) for p in (output / "evidence").rglob("*") if p.is_file()}
                       | {"filtering-evidence.json":final.sha(output / "filtering-evidence.json")}))
    (output / "quality").mkdir()
    with (sqlite3.connect((parent / "quality/checkpoints.sqlite").as_uri()+"?mode=ro",uri=True) as src,
          sqlite3.connect(output / "quality/checkpoints.sqlite") as dst):
        src.backup(dst)
        cached = dst.execute("SELECT file,result FROM checked").fetchall()
        final.require(len(cached) == len(files), "Parent audit cache incomplete")
        for ordinal, payload in cached:
            value = json.loads(payload)
            final.require(value["sha256"] == files[ordinal]["sha256"], "Parent cached payload differs")
            value["path"] = files[ordinal]["path"]
            dst.execute("UPDATE checked SET result=? WHERE file=?", (json.dumps(value),ordinal))
        dst.commit()
    write(output / "quality/binding.json", dict(binding, manifest_sha256=final.sha(output / "corpus-manifest.json")))
    final.require(verify_parent(parent)[0] == prepared, "Parent changed during packaging")
    return descriptor


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("parent","bundle","output"):
        p.add_argument("--"+name, type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(build(args.parent,args.bundle,args.output),indent=2))

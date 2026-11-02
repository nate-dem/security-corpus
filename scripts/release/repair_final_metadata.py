"""Apply an explicit, hash-bound exclusion and metadata repair plan to a new copy.

Retained text, licenses, quality columns and source metadata are preserved.
In metadata-repair mode, only source_url on canonical schemeless YouTube URLs
and published_at on NVD rows change. NVD times must agree with raw source
timestamps. Exclusions-only plans leave every retained field unchanged.
Excluded rows and their metadata remain in evidence outside the training files.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import html
import json
import logging
from pathlib import Path
import re
import shutil
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq

from ingest.connectors.vulnerability.nvd import _parse_datetime
from scripts.release import finalize_local as final
from scripts.release.correct_licenses import write

LOG = logging.getLogger(__name__)
REVISION = "mechanical-cleanup-20261023"
WATCH_URL = re.compile(r"youtube\.com/watch\?v=[A-Za-z0-9_-]{11}\Z")


def transform(table, file, start, drops, *, repair_metadata=True, title_repairs=None):
    """Return retained rows and exact changes; reject stale exclusion/source data."""
    retained, excluded, changes = [], [], []
    urls = table["source_url"].to_pylist()
    stamps = table["published_at"].to_pylist()
    titles = table["title"].to_pylist()
    cols = ["source_id", "record_id", "content_hash", "content_length", "raw"]
    nvd_seen = False
    for index, row in enumerate(table.select(cols).to_pylist()):
        address = (file, start + index)
        proof = drops.get(address)
        if proof:
            final.require(
                all(row[k] == proof[k] for k in ("record_id", "content_hash")),
                "Exclusion identity/hash mismatch",
            )
            final.require(
                row["source_id"] == proof["source_id"]
                and row["content_length"] == proof["tokens"],
                "Exclusion source/tokens mismatch",
            )
            excluded.append(index)
            continue
        retained.append(index)
        edits = {}
        title_proof = (title_repairs or {}).get(address)
        if title_proof:
            final.require(all(row[k] == title_proof[k] for k in
                              ("source_id", "record_id", "content_hash")),
                          "Title repair identity/hash mismatch")
            final.require(titles[index] == title_proof["before"], "Title repair value changed")
            expected_title = html.unescape(titles[index])
            final.require(expected_title == title_proof["after"] and expected_title != titles[index],
                          "Title repair is not exact HTML entity decoding")
            edits["title"] = dict(before=titles[index], after=expected_title,
                                  basis="explicit_title_only_html_entity_decoding")
            titles[index] = expected_title
        if not repair_metadata:
            if edits:
                changes.append(dict(file=file, parent_row=start+index, record_id=row["record_id"],
                                    content_hash=row["content_hash"], source=row["source_id"], fields=edits))
            continue
        if row["source_id"] == "youtube-commons" and WATCH_URL.fullmatch(
            urls[index] or ""
        ):
            edits["source_url"] = dict(
                before=urls[index], after="https://" + urls[index]
            )
            urls[index] = edits["source_url"]["after"]
        if row["source_id"] == "nvd":
            nvd_seen = True
            raw = json.loads(row["raw"]) if isinstance(row["raw"], str) else row["raw"]
            final.require(
                isinstance(raw, dict)
                and raw.get("cve", {}).get("id")
                == row["record_id"].removeprefix("nvd:"),
                "NVD raw identity mismatch",
            )
            expected = _parse_datetime(raw["cve"].get("published"))
            previous = stamps[index]
            final.require(
                expected is not None and isinstance(previous, datetime),
                "Missing NVD source timestamp",
            )
            interpreted = (
                previous.replace(tzinfo=timezone.utc)
                if previous.utcoffset() is None
                else previous.astimezone(timezone.utc)
            )
            final.require(
                interpreted == expected, "Stored NVD timestamp differs from raw source"
            )
            if previous.utcoffset() is None:
                edits["published_at"] = dict(
                    before=previous.isoformat(),
                    after=expected.isoformat(),
                    basis="raw_nvd_published_utc",
                )
            stamps[index] = expected
        if edits:
            changes.append(
                dict(
                    file=file,
                    parent_row=start + index,
                    record_id=row["record_id"],
                    content_hash=row["content_hash"],
                    source=row["source_id"],
                    fields=edits,
                )
            )
    after = table
    if any("title" in c["fields"] for c in changes):
        field = table.schema.field("title")
        after = after.set_column(table.schema.get_field_index("title"), field,
                                 pa.array(titles, type=field.type))
    if any("source_url" in c["fields"] for c in changes):
        field = table.schema.field("source_url")
        after = after.set_column(
            table.schema.get_field_index("source_url"),
            field,
            pa.array(urls, type=field.type),
        )
    if nvd_seen:
        field = table.schema.field("published_at").with_type(
            pa.timestamp("us", tz="UTC")
        )
        after = after.set_column(
            table.schema.get_field_index("published_at"),
            field,
            pa.array(stamps, type=field.type),
        )
    after = after.take(pa.array(retained, type=pa.int64()))
    excluded_table = table.take(pa.array(excluded, type=pa.int64()))
    return after, excluded_table, changes


def load_plan(path, manifest_sha):
    plan = final.read(path)
    final.require(
        plan["approved"] is True and plan["parent_manifest_sha256"] == manifest_sha,
        "Plan not approved or wrong parent",
    )
    mode = plan.get("mode", "metadata-repair")
    final.require(mode in {"metadata-repair", "exclusions-only", "explicit-title-repair"}, "Unknown plan mode")
    final.require(
        isinstance(plan["revision"], str)
        and re.fullmatch(r"[a-z][a-z0-9-]{0,79}", plan["revision"])
        and (mode != "metadata-repair" or plan["revision"] == REVISION),
        "Unexpected revision",
    )
    drops = {}
    for proof in plan["exclusions"]:
        key = (proof["file"], proof["row"])
        final.require(key not in drops, "Duplicate exclusion address")
        drops[key] = proof
    final.require(
        len(drops) == plan["expected"]["excluded_records"], "Exclusion count mismatch"
    )
    final.require(
        sum(p["tokens"] for p in drops.values()) == plan["expected"]["excluded_tokens"],
        "Exclusion token count mismatch",
    )
    return plan, drops


def prepare(parent, plan_path, output):
    parent, plan_path, output = (p.resolve() for p in (parent, plan_path, output))
    final.require(not output.exists(), "Use a new revision directory")
    final.require(not output.is_relative_to(parent), "Output must be outside parent")
    prepared, manifest = final.verify_prepared(parent)
    marker = final.read(parent / "final-corpus-complete.json")
    for key, name in (
        ("manifest_sha256", "corpus-manifest.json"),
        ("prepared_sha256", "final-prepared.json"),
        ("quality_summary_sha256", "quality/summary.json"),
    ):
        final.require(
            final.sha(parent / name) == marker[key], "Parent completion binding differs"
        )
    final.require(marker["integrity_passed"], "Parent integrity failed")
    plan, drops = load_plan(plan_path, prepared["manifest_sha256"])
    revision_name = plan["revision"]
    repair_metadata = plan.get("mode", "metadata-repair") == "metadata-repair"
    title_repairs = {(p["file"], p["row"]): p for p in plan.get("title_repairs", [])}
    final.require(len(title_repairs) == len(plan.get("title_repairs", [])), "Duplicate title repair")
    final.require(not title_repairs or plan["mode"] == "explicit-title-repair", "Wrong title mode")
    final.require(not (set(title_repairs) & set(drops)), "Title repair also excluded")
    final.require(len(title_repairs) == plan["expected"].get("title", 0), "Title repair count differs")
    binding = final.read(parent / "quality/binding.json")
    final.require(
        binding["manifest_sha256"] == prepared["manifest_sha256"]
        and binding["code_sha256"] == final.sha(Path(final.__file__)),
        "Parent audit cache binding differs",
    )
    final.require(
        not (parent / "evidence" / revision_name).exists(), "Revision already applied"
    )
    for name, digest in plan["supporting_evidence_sha256"].items():
        final.require(
            final.sha(final.contained(plan_path.parent, name)) == digest,
            "Plan evidence differs",
        )
    output.mkdir()
    (output / "records").mkdir()
    shutil.copytree(
        parent / "evidence", output / "evidence", copy_function=final.link_or_copy
    )
    evidence = output / "evidence" / revision_name
    evidence.mkdir()
    shutil.copytree(plan_path.parent, evidence / "approved-plan")
    for name in (
        "corpus-manifest.json",
        "final-prepared.json",
        "final-corpus-complete.json",
        "schema.json",
    ):
        shutil.copyfile(parent / name, evidence / ("parent-" + name))
    shutil.copyfile(
        parent / "quality/summary.json", evidence / "parent-quality-summary.json"
    )
    shutil.copyfile(parent / "schema.json", output / "schema.json")
    changes_db = evidence / "changes.sqlite"
    counts, changed_files, files, seen_drops = Counter(), [], [], set()
    with sqlite3.connect(changes_db) as db:
        db.execute(
            "CREATE TABLE changes(file INTEGER,parent_row INTEGER,note TEXT,PRIMARY KEY(file,parent_row))"
        )
        for ordinal, entry in enumerate(manifest["files"]):
            source = final.contained(parent, entry["relative_path"])
            target = final.contained(output, entry["relative_path"])
            final.require(
                final.sha(source) == entry["sha256"], "Parent payload differs"
            )
            excluded_rows = sorted(row for file, row in drops if file == ordinal)
            with pq.ParquetFile(source) as pf:
                sources = (
                    set(pf.read(columns=["source_id"])["source_id"].to_pylist())
                    if repair_metadata else set()
                )
                url_fix = False
                if "youtube-commons" in sources:
                    url_fix = any(
                        WATCH_URL.fullmatch(v or "")
                        for v in pf.read(columns=["source_url"])[
                            "source_url"
                        ].to_pylist()
                    )
                affected = bool(excluded_rows or url_fix or "nvd" in sources
                                or any(file == ordinal for file, _ in title_repairs))
                removed = removed_tokens = 0
                if affected:
                    changed_files.append(ordinal)
                    writer, excluded_writer, offset = None, None, 0
                    try:
                        for batch in pf.iter_batches(batch_size=512):
                            before = pa.Table.from_batches([batch])
                            after, excluded, changes = transform(
                                before, ordinal, offset, drops,
                                repair_metadata=repair_metadata,
                                title_repairs=title_repairs,
                            )
                            if writer is None:
                                writer = pq.ParquetWriter(
                                    target, after.schema, compression="zstd"
                                )
                            if len(after):
                                writer.write_table(after)
                            if len(excluded):
                                if excluded_writer is None:
                                    excluded_writer = pq.ParquetWriter(
                                        evidence / f"excluded-{ordinal:06d}.parquet",
                                        before.schema,
                                        compression="zstd",
                                    )
                                excluded_writer.write_table(excluded)
                                removed += len(excluded)
                                removed_tokens += sum(
                                    excluded["content_length"].to_pylist()
                                )
                                seen_drops.update(
                                    (ordinal, i)
                                    for i in excluded_rows
                                    if offset <= i < offset + len(before)
                                )
                            for note in changes:
                                db.execute(
                                    "INSERT INTO changes VALUES (?,?,?)",
                                    (ordinal, note["parent_row"], json.dumps(note)),
                                )
                                counts.update(note["fields"].keys())
                            offset += len(before)
                    finally:
                        if writer is not None:
                            writer.close()
                        if excluded_writer is not None:
                            excluded_writer.close()
                    # Each write is one row group. Compare all retained fields
                    # with independently transformed parent batches after reading.
                    with pq.ParquetFile(target) as checked:
                        offset = group = 0
                        for batch in pf.iter_batches(batch_size=512):
                            expected, _, _ = transform(
                                pa.Table.from_batches([batch]), ordinal, offset, drops,
                                repair_metadata=repair_metadata,
                                title_repairs=title_repairs,
                            )
                            if len(expected):
                                final.require(
                                    expected.equals(
                                        checked.read_row_group(group),
                                        check_metadata=True,
                                    ),
                                    "Roundtrip changed retained fields",
                                )
                                group += 1
                            offset += len(batch)
                        final.require(
                            group == checked.num_row_groups, "Extra output row groups"
                        )
                else:
                    final.link_or_copy(source, target)
            final.require(
                final.sha(source) == entry["sha256"], "Parent changed during revision"
            )
            files.append(
                dict(
                    entry,
                    path=str(target),
                    sha256=final.sha(target),
                    bytes=target.stat().st_size,
                    records=entry["records"] - removed,
                    tokens=entry["tokens"] - removed_tokens,
                    parent_sha256=entry["sha256"],
                    excluded_parent_rows=excluded_rows,
                )
            )
            counts.update(excluded_records=removed, excluded_tokens=removed_tokens)
            if ordinal % 250 == 0 or ordinal + 1 == len(manifest["files"]):
                db.commit()
                LOG.info(
                    "Revision %s/%s files; changes=%s",
                    ordinal + 1,
                    len(manifest["files"]),
                    dict(counts),
                )
        db.commit()
        final.require(seen_drops == set(drops), "Unapplied exclusion targets")
        for key, value in plan["expected"].items():
            final.require(
                counts[key] == value,
                f"Unexpected revision count: {key}={counts[key]} vs {value}",
            )
        seen_metadata, seen_edits, output_rows, parent_rows = (
            set(),
            0,
            Counter(),
            Counter(),
        )
        with (
            gzip.open(parent / "record-metadata.jsonl.gz", "rt") as src,
            gzip.open(output / "record-metadata.jsonl.gz", "wt") as dst,
            gzip.open(evidence / "excluded-metadata.jsonl.gz", "wt") as excluded_out,
        ):
            for line in src:
                note = json.loads(line)
                ordinal, row = note["file"], note["output_row_index"]
                final.require(
                    row == parent_rows[ordinal], "Parent metadata row sequence differs"
                )
                parent_rows[ordinal] += 1
                proof = drops.get((ordinal, row))
                if proof:
                    final.require(
                        note["record_id"] == proof["record_id"]
                        and note["content_hash"] == proof["content_hash"],
                        "Excluded metadata identity differs",
                    )
                    excluded_out.write(line)
                    seen_metadata.add((ordinal, row))
                    continue
                edit = db.execute(
                    "SELECT note FROM changes WHERE file=? AND parent_row=?",
                    (ordinal, row),
                ).fetchone()
                if edit:
                    change = json.loads(edit[0])
                    final.require(
                        note["record_id"] == change["record_id"]
                        and note["content_hash"] == change["content_hash"],
                        "Changed metadata identity differs",
                    )
                    note.setdefault("canonical_metadata_revisions", []).append(
                        dict(
                            revision=revision_name,
                            fields=change["fields"],
                            parent_manifest_sha256=prepared["manifest_sha256"],
                        )
                    )
                    seen_edits += 1
                if files[ordinal]["excluded_parent_rows"]:
                    note["previous_revision_output_row_index"] = row
                note["output_row_index"] = output_rows[ordinal]
                output_rows[ordinal] += 1
                dst.write(json.dumps(note) + "\n")
        final.require(seen_metadata == set(drops), "Missing excluded metadata")
        final.require(
            seen_edits == db.execute("SELECT count(*) FROM changes").fetchone()[0],
            "Missing changed metadata",
        )
        final.require(
            all(
                parent_rows[i] == e["records"] for i, e in enumerate(manifest["files"])
            ),
            "Parent metadata count differs",
        )
        final.require(
            all(output_rows[i] == e["records"] for i, e in enumerate(files)),
            "Output metadata count differs",
        )
    totals = {k: sum(e[k] for e in files) for k in ("records", "tokens", "bytes")}
    revision = dict(
        revision=revision_name,
        mode=plan.get("mode", "metadata-repair"),
        parent_manifest_sha256=prepared["manifest_sha256"],
        changed_files=changed_files,
        changes=dict(counts),
        totals=totals,
        retained_text_licenses_quality_columns_unchanged=True,
        schema_compatibility_assessed_by_this_revision=False,
        release_ready=False,
    )
    write(evidence / "revision.json", revision)
    write(
        output / "corpus-manifest.json",
        dict(
            manifest,
            files=files,
            totals=totals,
            parent_manifest_sha256=prepared["manifest_sha256"],
        ),
    )
    write(
        output / "final-prepared.json",
        dict(
            prepared,
            manifest_sha256=final.sha(output / "corpus-manifest.json"),
            record_metadata_sha256=final.sha(output / "record-metadata.jsonl.gz"),
            schema_sha256=final.sha(output / "schema.json"),
            code_sha256=final.sha(Path(__file__)),
            totals=totals,
            excluded_records=prepared.get("excluded_records", 0)
            + counts["excluded_records"],
            excluded_tokens=prepared.get("excluded_tokens", 0)
            + counts["excluded_tokens"],
            evidence_sha256={
                str(p.relative_to(output)): final.sha(p)
                for p in (output / "evidence").rglob("*")
                if p.is_file()
            },
        ),
    )
    # Cache reuse requires byte-identical payloads. Every changed shard is
    # independently rehashed and retokenized by final.audit().
    (output / "quality").mkdir()
    with (
        sqlite3.connect(
            (parent / "quality/checkpoints.sqlite").as_uri() + "?mode=ro", uri=True
        ) as src,
        sqlite3.connect(output / "quality/checkpoints.sqlite") as dst,
    ):
        src.backup(dst)
        for ordinal in changed_files:
            dst.execute("DELETE FROM checked WHERE file=?", (ordinal,))
            dst.execute("DELETE FROM identities WHERE file=?", (ordinal,))
        for ordinal, payload in dst.execute(
            "SELECT file,result FROM checked"
        ).fetchall():
            value = json.loads(payload)
            final.require(
                value["sha256"] == files[ordinal]["sha256"],
                "Unchanged cache payload differs",
            )
            value["path"] = files[ordinal]["path"]
            dst.execute(
                "UPDATE checked SET result=? WHERE file=?", (json.dumps(value), ordinal)
            )
        dst.commit()
    write(
        output / "quality/binding.json",
        dict(binding, manifest_sha256=final.sha(output / "corpus-manifest.json")),
    )
    final.require(
        final.verify_prepared(parent)[0] == prepared, "Parent evidence changed"
    )
    return revision


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    print(json.dumps(prepare(a.parent, a.plan, a.output), indent=2))

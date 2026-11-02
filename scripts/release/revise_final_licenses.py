"""Create an auditable license-only revision of a completed local corpus.

SO/SE's unresolved family label becomes CC-BY-SA at the researcher's request.
An explicit arXiv review supplies row/hash-bound upstream declarations. All
other columns are compared after a Parquet round trip. Unchanged files are
hardlinked; the parent remains valid. Only unchanged-file audit checkpoints
are reused; changed files receive the existing full token/hash audit.

--source-terms applies the researcher's 23 October descriptive labels for
flaws.cloud and Reddit, retaining unresolved permission evidence separately.
"""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import shutil
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.release import finalize_local as final
from scripts.release.correct_licenses import write
from scripts.release.review_arxiv_license_pages import parse_page


OLD_QA = "CC-BY-SA (version unknown)"
SOURCE_TERMS_REVISION = "source-terms-labels-20261023"


def source_terms_label(source):
    if source == "cloudtrail-flaws":
        return (
            "Public",
            "public_availability_label_no_redistribution_license_established",
        )
    if source == "reddit" or source.startswith("reddit-"):
        return (
            "Reddit Terms of Service",
            "platform_terms_label_no_content_redistribution_license_established",
        )
    return None


def is_qa(source):
    return source == "stackoverflow" or source.startswith("stackexchange-")


def revise_batch(table, ordinal, start, papers, source_terms=False):
    """Change explicit target licenses, preserving every other field and row."""
    labels = table["license"].to_pylist()
    rows = table.select(["source_id", "record_id", "content_hash"]).to_pylist()
    changes = []
    for offset, row in enumerate(rows):
        address = (ordinal, start + offset)
        old = labels[offset]
        paper = papers.get(address)
        if source_terms:
            rule = source_terms_label(row["source_id"])
            if rule is None:
                continue
            final.require(old == "NOASSERTION", "Unexpected prior source-terms label")
            label, basis = rule
            paper = dict(
                kind="researcher_requested_descriptive_source_label",
                request_date="2026-10-23",
                license_verification="unresolved",
                public_release_approved=False,
                public_domain_asserted=False,
            )
        elif is_qa(row["source_id"]) and old == OLD_QA:
            label, basis = "CC-BY-SA", "contribution_versions_unresolved"
        elif paper:
            final.require(
                row["source_id"] == "arxiv"
                and old == "NOASSERTION"
                and row["record_id"] == paper["record_id"]
                and row["content_hash"] == paper["content_hash"],
                "arXiv target identity or previous license changed",
            )
            label, basis = paper["license"], paper["license_basis"]
        else:
            continue
        labels[offset] = label
        changes.append(
            dict(
                file=ordinal,
                row=start + offset,
                **row,
                previous_license=old,
                license=label,
                license_basis=basis,
                evidence=paper if paper else None,
            )
        )
    if not changes:
        return table, changes
    return table.set_column(
        table.schema.get_field_index("license"),
        table.schema.field("license"),
        pa.array(labels, type=table.schema.field("license").type),
    ), changes


def prepare(parent, review, output, source_terms=False):
    parent, review, output = (p.resolve() for p in (parent, review, output))
    final.require(not output.exists(), "Use a new revision directory")
    # Copied parent evidence may be hardlinked. Never overwrite an inherited
    # evidence name when applying this workflow to an already-revised parent.
    reserved = (
        (SOURCE_TERMS_REVISION,)
        if source_terms
        else (
            "license-review",
            "license-changes.jsonl",
            "license-revision.json",
            "parent-corpus-manifest.json",
            "parent-final-prepared.json",
            "parent-final-corpus-complete.json",
            "parent-quality-summary.json",
        )
    )
    final.require(
        not any((parent / "evidence" / name).exists() for name in reserved),
        "Parent already contains revision evidence; use a separate revision plan",
    )
    prepared, manifest = final.verify_prepared(parent)
    completed = final.read(parent / "final-corpus-complete.json")
    for key, name in (
        ("manifest_sha256", "corpus-manifest.json"),
        ("prepared_sha256", "final-prepared.json"),
        ("quality_summary_sha256", "quality/summary.json"),
    ):
        final.require(
            final.sha(parent / name) == completed[key],
            "Parent completion binding changed",
        )
    quality = final.read(parent / "quality/summary.json")
    final.require(
        completed["integrity_passed"] and quality["integrity_passed"],
        "Parent audit failed",
    )
    binding = final.read(parent / "quality/binding.json")
    final.require(
        binding["manifest_sha256"] == prepared["manifest_sha256"],
        "Parent cache binding changed",
    )
    final.require(
        binding["code_sha256"] == final.sha(Path(final.__file__)), "Audit code changed"
    )
    papers = {}
    if source_terms:
        request = final.read(review / "researcher-request.json")
        final.require(
            request["labels"]
            == {"cloudtrail-flaws": "Public", "reddit": "Reddit Terms of Service"}
            and request["permissions_resolved"] is False,
            "Source-label request does not match this revision",
        )
        paper_lines = []
    else:
        review_summary = final.read(review / "arxiv-official/summary.json")
        final.require(
            review_summary["checked"] == review_summary["requested"],
            "Incomplete paper review",
        )
        paper_lines = (review / "arxiv-official/records.jsonl").read_text().splitlines()
    for line in paper_lines:
        paper = json.loads(line)
        if paper["status"] != "declaration_found":
            continue
        page = Path(paper["evidence_path"])
        final.require(page.is_relative_to(review), "Evidence outside review")
        final.require(final.sha(page) == paper["sha256"], "Paper evidence changed")
        parsed = parse_page(
            page.read_bytes(), paper["record_id"].removeprefix("arxiv:")
        )
        final.require(
            all(paper[k] == v for k, v in parsed.items()),
            "Paper declaration differs from page",
        )
        paper["evidence_path"] = "evidence/license-review/" + str(
            page.relative_to(review)
        )
        address = (paper["file"], paper["row"])
        final.require(address not in papers, "Duplicate paper address")
        papers[address] = paper
    output.mkdir()
    (output / "records").mkdir()
    shutil.copytree(
        parent / "evidence", output / "evidence", copy_function=final.link_or_copy
    )
    revision_evidence = output / "evidence"
    if source_terms:
        revision_evidence = revision_evidence / SOURCE_TERMS_REVISION
        revision_evidence.mkdir()
    shutil.copytree(
        review, revision_evidence / "license-review", copy_function=shutil.copyfile
    )
    for name in (
        "corpus-manifest.json",
        "final-prepared.json",
        "final-corpus-complete.json",
    ):
        shutil.copyfile(parent / name, revision_evidence / ("parent-" + name))
    shutil.copyfile(
        parent / "quality/summary.json",
        revision_evidence / "parent-quality-summary.json",
    )
    shutil.copyfile(parent / "schema.json", output / "schema.json")
    if source_terms:
        schema = final.read(output / "schema.json")
        schema["license"] = (
            "Recorded declaration, descriptive source-terms label, or unresolved marker; "
            "not redistribution approval. Public means publicly available, not public domain. "
            "Reddit Terms of Service names platform terms; it does not assert an open content license."
        )
        write(output / "schema.json", schema)
    files, changes, changed_files = [], {}, []
    for ordinal, entry in enumerate(manifest["files"]):
        source = Path(entry["path"])
        final.require(final.sha(source) == entry["sha256"], "Parent payload changed")
        target = final.contained(output, entry["relative_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        with pq.ParquetFile(source) as pf:
            sources = set(pf.read(columns=["source_id"])["source_id"].to_pylist())
            affected = (
                any(source_terms_label(s) for s in sources)
                if source_terms
                else any(is_qa(s) for s in sources)
                or any(f == ordinal for f, _ in papers)
            )
            if affected:
                offset = 0
                with pq.ParquetWriter(
                    target, pf.schema_arrow, compression="zstd"
                ) as writer:
                    for batch in pf.iter_batches(batch_size=512):
                        before = pa.Table.from_batches([batch])
                        after, notes = revise_batch(
                            before, ordinal, offset, papers, source_terms
                        )
                        writer.write_table(after)
                        for note in notes:
                            changes[(ordinal, note["row"])] = note
                        offset += len(before)
                # Match bounded batches independently of saved row-group boundaries.
                with pq.ParquetFile(target) as checked:
                    before_batches = pf.iter_batches(batch_size=512)
                    after_batches = checked.iter_batches(batch_size=512)
                    offset = 0
                    for before, after in zip(
                        before_batches, after_batches, strict=True
                    ):
                        expected, _ = revise_batch(
                            pa.Table.from_batches([before]),
                            ordinal,
                            offset,
                            papers,
                            source_terms,
                        )
                        final.require(
                            expected.equals(
                                pa.Table.from_batches([after]), check_metadata=True
                            ),
                            "License revision roundtrip changed other columns",
                        )
                        offset += len(before)
                changed_files.append(ordinal)
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
                parent_sha256=entry["sha256"],
            )
        )
    final.require(set(papers) <= set(changes), "Some paper targets were not applied")
    with (revision_evidence / "license-changes.jsonl").open("w") as f:
        for note in changes.values():
            f.write(json.dumps(note) + "\n")
    seen = set()
    with (
        gzip.open(parent / "record-metadata.jsonl.gz", "rt") as src,
        gzip.open(output / "record-metadata.jsonl.gz", "wt") as dst,
    ):
        for line in src:
            note = json.loads(line)
            key = (note["file"], note["output_row_index"])
            change = changes.get(key)
            if change:
                final.require(
                    note["license"] == change["previous_license"]
                    and note["record_id"] == change["record_id"]
                    and note["content_hash"] == change["content_hash"],
                    "Metadata target mismatch",
                )
                note.update(
                    previous_revision_license=note["license"],
                    previous_revision_license_basis=note.get("license_basis"),
                    license=change["license"],
                    license_basis=change["license_basis"],
                    license_revision_evidence=change["evidence"],
                )
                seen.add(key)
            dst.write(json.dumps(note) + "\n")
    final.require(seen == set(changes), "Incomplete metadata changes")
    totals = dict(manifest["totals"], bytes=sum(e["bytes"] for e in files))
    write(
        output / "corpus-manifest.json",
        dict(
            manifest,
            files=files,
            totals=totals,
            parent_manifest_sha256=prepared["manifest_sha256"],
        ),
    )
    revision = dict(
        parent=str(parent),
        parent_manifest_sha256=prepared["manifest_sha256"],
        changed_records=len(changes),
        changed_files=changed_files,
        changes_by_source=dict(Counter(n["source_id"] for n in changes.values())),
        changes_by_license=dict(Counter(n["license"] for n in changes.values())),
        all_non_license_columns_compared_unchanged=True,
        exact_arxiv_downloaded_version_verified=False,
        qa_exact_versions_resolved=False,
        release_ready=False,
        descriptive_source_labels=source_terms,
    )
    write(revision_evidence / "license-revision.json", revision)
    write(
        output / "final-prepared.json",
        dict(
            prepared,
            totals=totals,
            manifest_sha256=final.sha(output / "corpus-manifest.json"),
            record_metadata_sha256=final.sha(output / "record-metadata.jsonl.gz"),
            schema_sha256=final.sha(output / "schema.json"),
            evidence_sha256={
                str(p.relative_to(output)): final.sha(p)
                for p in (output / "evidence").rglob("*")
                if p.is_file()
            },
            code_sha256=final.sha(Path(__file__)),
        ),
    )
    # Transfer only cache entries for byte-identical files. Changed files will
    # have their text hashes and token counts recomputed by final.audit().
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
                "Transferred cache hash differs",
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-terms", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.parent, args.review, args.output, args.source_terms), indent=2
        )
    )

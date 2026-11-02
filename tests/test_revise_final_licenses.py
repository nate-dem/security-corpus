from datetime import datetime, timezone
import gzip
import json
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ingest.utils import compute_content_hash, compute_token_count
from scripts.release import finalize_local as final
from scripts.release.correct_licenses import write
from scripts.release.review_arxiv_license_pages import parse_page, run
from scripts.release.revise_final_licenses import OLD_QA, prepare, revise_batch


def article(identifier="1506.07955"):
    return f'''<html><head>
    <meta name="citation_arxiv_id" content="{identifier}">
    <meta name="citation_title" content="Example paper">
    <meta property="og:url" content="https://arxiv.org/abs/{identifier}v2">
    </head><body><div class="submission-history">[v1] [v2]</div>
    <div class="abs-license"><a href="https://creativecommons.org/publicdomain/zero/1.0/">license</a></div>
    <footer><a href="https://creativecommons.org/licenses/by/4.0/">metadata license</a></footer>
    </body></html>'''.encode()


def test_article_identity_and_license_scope():
    parsed = parse_page(article(), "1506.07955")
    assert parsed["license"] == "CC0-1.0"
    assert parsed["versions_in_history"] == ["1", "2"]
    assert parsed["article_version"] == "1506.07955v2"
    assert parsed["paper_version_binding_established"] is False
    with pytest.raises(ValueError, match="identity mismatch"):
        parse_page(article(), "1506.07956")
    with pytest.raises(ValueError, match="one article license"):
        parse_page(article().replace(b"abs-license", b"other"), "1506.07955")


def test_explicit_version_cannot_reuse_latest_version_page(tmp_path):
    output = tmp_path / "review"
    pages = output / "pages"
    pages.mkdir(parents=True)
    page = pages / "1506.07955v1.html"
    page.write_bytes(article())  # The fixture declares v2, not requested v1.
    write(pages / "1506.07955v1.json", dict(sha256=final.sha(page)))
    targets = tmp_path / "targets.jsonl"
    targets.write_text(
        json.dumps(dict(record_id="arxiv:1506.07955", review_version=1)) + "\n"
    )
    result = run(targets, output)
    assert result["statuses"] == {"unresolved": 1}
    record = json.loads((output / "records.jsonl").read_text())
    assert record["error"] == "Requested version differs from returned version"


def test_prior_revision_evidence_is_not_overwritten(tmp_path):
    parent = tmp_path / "parent"
    (parent / "evidence").mkdir(parents=True)
    old = parent / "evidence/license-changes.jsonl"
    old.write_text("original evidence")
    with pytest.raises(ValueError, match="already contains revision evidence"):
        prepare(parent, tmp_path / "review", tmp_path / "next")
    assert old.read_text() == "original evidence"


def record(source, identifier, license_value):
    content = f"Security research example for {source} record {identifier}."
    return dict(
        source_id=source,
        source_record_id=identifier,
        record_id=f"{source}:{identifier}",
        content=content,
        content_hash=compute_content_hash(content),
        content_length=compute_token_count(content),
        title="Example",
        license=license_value,
        ingested_at=datetime.now(timezone.utc),
        published_at=None,
        source_url=None,
        raw=None,
        qwen_quality=4.25,
    )


def test_only_unresolved_qa_label_changes():
    rows = [
        record("stackoverflow", "1", OLD_QA),
        record("stackexchange-infosec", "2", "CC-BY-SA-4.0"),
        record("reddit-security", "3", "NOASSERTION"),
    ]
    before = pa.Table.from_pylist(rows)
    after, notes = revise_batch(before, 0, 0, {})
    assert after["license"].to_pylist() == ["CC-BY-SA", "CC-BY-SA-4.0", "NOASSERTION"]
    assert len(notes) == 1
    assert notes[0]["license_basis"] == "contribution_versions_unresolved"
    cols = [c for c in before.column_names if c != "license"]
    assert after.select(cols).equals(before.select(cols))


def test_arxiv_must_match_physical_row_hash():
    row = record("arxiv", "1506.07955", "NOASSERTION")
    proof = dict(
        record_id=row["record_id"],
        content_hash="wrong",
        license="CC0-1.0",
        license_basis="official_page",
    )
    with pytest.raises(ValueError, match="target identity"):
        revise_batch(pa.Table.from_pylist([row]), 0, 0, {(0, 0): proof})


def test_source_terms_labels_do_not_clear_permissions_or_change_other_sources():
    rows = [
        record("cloudtrail-flaws", "1", "NOASSERTION"),
        record("reddit-security", "2", "NOASSERTION"),
        record("stackoverflow", "3", "CC-BY-SA"),
        record("arxiv", "4", "CC0-1.0"),
    ]
    before = pa.Table.from_pylist(rows)
    after, notes = revise_batch(before, 0, 0, {}, source_terms=True)
    assert after["license"].to_pylist() == [
        "Public",
        "Reddit Terms of Service",
        "CC-BY-SA",
        "CC0-1.0",
    ]
    assert len(notes) == 2
    for note in notes:
        assert note["evidence"]["license_verification"] == "unresolved"
        assert note["evidence"]["public_domain_asserted"] is False
        assert note["evidence"]["public_release_approved"] is False
    columns = [c for c in before.column_names if c != "license"]
    assert before.select(columns).equals(after.select(columns), check_metadata=True)


def test_completed_revision_preserves_parent_and_reuses_only_identical_files(tmp_path):
    parent, review, output = (tmp_path / n for n in ("parent", "review", "revision"))
    (parent / "records").mkdir(parents=True)
    (parent / "evidence").mkdir()
    rows = [
        record("stackoverflow", "1", OLD_QA),
        record("arxiv", "1506.07955", "NOASSERTION"),
    ]
    groups = [
        rows,
        [record("example", "unchanged", "CC-BY-4.0")],
        [
            record("cloudtrail-flaws", "log-1", "NOASSERTION"),
            record("reddit-security", "thread-1", "NOASSERTION"),
        ],
    ]
    files, notes = [], []
    for ordinal, group in enumerate(groups):
        relative = f"records/{ordinal:06d}.parquet"
        path = parent / relative
        pq.write_table(pa.Table.from_pylist(group), path)
        files.append(
            dict(
                path=str(path),
                relative_path=relative,
                sha256=final.sha(path),
                records=len(group),
                tokens=sum(r["content_length"] for r in group),
                bytes=path.stat().st_size,
            )
        )
        notes.extend(
            dict(
                file=ordinal,
                output_row_index=i,
                source=r["source_id"],
                record_id=r["record_id"],
                content_hash=r["content_hash"],
                license=r["license"],
                license_basis="old",
            )
            for i, r in enumerate(group)
        )
    totals = {k: sum(f[k] for f in files) for k in ("records", "tokens", "bytes")}
    write(parent / "corpus-manifest.json", dict(files=files, totals=totals))
    write(parent / "schema.json", dict(canonical_fields=final.BASE_FIELDS))
    with gzip.open(parent / "record-metadata.jsonl.gz", "wt") as f:
        for note in notes:
            f.write(json.dumps(note) + "\n")
    prepared = dict(
        version=final.VERSION,
        prepared=True,
        totals=totals,
        evidence_sha256={},
        retained_text_and_quality_columns_unchanged=True,
    )
    for key, name in (
        ("manifest_sha256", "corpus-manifest.json"),
        ("schema_sha256", "schema.json"),
        ("record_metadata_sha256", "record-metadata.jsonl.gz"),
    ):
        prepared[key] = final.sha(parent / name)
    write(parent / "final-prepared.json", prepared)
    assert final.audit(parent, workers=1)["integrity_passed"]
    parent_hashes = {p: final.sha(p) for p in parent.rglob("*") if p.is_file()}
    official = review / "arxiv-official"
    (official / "pages").mkdir(parents=True)
    page = official / "pages/1506.07955.html"
    page.write_bytes(article())
    paper = dict(
        record_id=rows[1]["record_id"],
        content_hash=rows[1]["content_hash"],
        file=0,
        row=1,
        status="declaration_found",
        sha256=final.sha(page),
        evidence_path=str(page),
        **parse_page(article(), "1506.07955"),
    )
    (official / "records.jsonl").write_text(json.dumps(paper) + "\n")
    write(official / "summary.json", dict(checked=1, requested=1))
    revision = prepare(parent, review, output)
    assert revision["changed_records"] == 2
    assert revision["changed_files"] == [0]
    with sqlite3.connect(output / "quality/checkpoints.sqlite") as db:
        assert db.execute("SELECT file FROM checked").fetchall() == [(1,), (2,)]
    report = final.audit(output, workers=1)
    assert report["integrity_passed"]
    assert report["licenses"] == {
        "CC-BY-SA": 1,
        "CC0-1.0": 1,
        "CC-BY-4.0": 1,
        "NOASSERTION": 2,
    }
    assert report["totals"]["tokens"] == totals["tokens"]
    assert parent_hashes == {p: final.sha(p) for p in parent_hashes}
    assert final.read(output / "final-corpus-complete.json")["release_ready"] is False
    # A subsequent source-label revision must preserve all prior, hardlinked
    # evidence and the earlier arXiv/QA labels.
    previous_hashes = {p: final.sha(p) for p in output.rglob("*") if p.is_file()}
    terms_review = tmp_path / "source-terms-review"
    terms_review.mkdir()
    write(
        terms_review / "researcher-request.json",
        dict(
            labels={"cloudtrail-flaws": "Public", "reddit": "Reddit Terms of Service"},
            permissions_resolved=False,
        ),
    )
    terms_output = tmp_path / "terms-final"
    change = prepare(output, terms_review, terms_output, source_terms=True)
    assert change["changed_files"] == [2]
    assert change["changes_by_license"] == {"Public": 1, "Reddit Terms of Service": 1}
    result = final.audit(terms_output, workers=1)
    assert result["integrity_passed"]
    assert result["totals"]["tokens"] == totals["tokens"]
    assert result["licenses"] == {
        "CC-BY-SA": 1,
        "CC0-1.0": 1,
        "CC-BY-4.0": 1,
        "Public": 1,
        "Reddit Terms of Service": 1,
    }
    assert previous_hashes == {p: final.sha(p) for p in previous_hashes}
    with gzip.open(terms_output / "record-metadata.jsonl.gz", "rt") as f:
        terms_notes = [json.loads(line) for line in f if '"file": 2' in line]
    assert len(terms_notes) == 2
    assert all(n["previous_revision_license_basis"] == "old" for n in terms_notes)
    assert all(
        n["license_revision_evidence"]["license_verification"] == "unresolved"
        for n in terms_notes
    )

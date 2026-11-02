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
from scripts.release.repair_final_metadata import REVISION, prepare, transform


def record(source, identifier, **extra):
    text = f"Security research example {source} {identifier}."
    return dict(
        source_id=source,
        source_record_id=identifier,
        record_id=f"{source}:{identifier}",
        content=text,
        content_hash=compute_content_hash(text),
        content_length=compute_token_count(text),
        title="Example",
        license="CC-BY-4.0",
        ingested_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        published_at=None,
        source_url=None,
        raw=None,
        qwen_quality=3,
        **extra,
    )


def proof(row, file=0, index=0):
    return dict(
        file=file,
        row=index,
        source_id=row["source_id"],
        record_id=row["record_id"],
        content_hash=row["content_hash"],
        tokens=row["content_length"],
    )


def test_hash_bound_drop_does_not_remove_wrong_row():
    row = record("arxiv", "1")
    target = proof(row)
    target["content_hash"] = "wrong"
    with pytest.raises(ValueError, match="identity/hash"):
        transform(pa.Table.from_pylist([row]), 0, 0, {(0, 0): target})


def test_only_canonical_youtube_watch_urls_change():
    rows = [record("youtube-commons", str(i)) for i in range(4)]
    urls = [
        "youtube.com/watch?v=abcdefghijk",
        "https://youtube.com/watch?v=abcdefghijk",
        "youtube.com/watch?v=bad",
        "youtube.com.evil/watch?v=abcdefghijk",
    ]
    for r, u in zip(rows, urls):
        r["source_url"] = u
    before = pa.Table.from_pylist(rows)
    after, excluded, notes = transform(before, 0, 0, {})
    assert after["source_url"].to_pylist() == ["https://" + urls[0], *urls[1:]]
    assert not len(excluded) and len(notes) == 1
    cols = [c for c in before.column_names if c != "source_url"]
    assert before.select(cols).equals(after.select(cols))


def test_explicit_title_repair_preserves_content_and_rejects_other_edits():
    row = record("reddit-netsec", "example")
    row["title"] = "RouterOS &amp; inetutils &lt;= 1.9.4"
    row["content"] = "# RouterOS &amp; inetutils &lt;= 1.9.4\nCode: &lt;"
    before = pa.Table.from_pylist([row])
    target = dict(proof(row), before=row["title"], after="RouterOS & inetutils <= 1.9.4")
    after, excluded, notes = transform(before, 0, 0, {}, repair_metadata=False,
                                       title_repairs={(0, 0): target})
    assert not len(excluded) and len(notes) == 1
    assert after["title"].to_pylist() == [target["after"]]
    columns = [c for c in before.column_names if c != "title"]
    assert after.select(columns).equals(before.select(columns))
    target["after"] = "new generated title"
    with pytest.raises(ValueError, match="exact HTML"):
        transform(before, 0, 0, {}, repair_metadata=False, title_repairs={(0, 0):target})


def nvd_record():
    r = record("nvd", "CVE-2020-1234")
    r.update(
        published_at=datetime(2020, 1, 1, 4, 0),
        raw=json.dumps(
            {"cve": {"id": "CVE-2020-1234", "published": "2020-01-01T04:00:00.000"}}
        ),
    )
    return r


def test_nvd_timestamp_requires_raw_identity_and_value():
    row = nvd_record()
    after, _, notes = transform(pa.Table.from_pylist([row]), 0, 0, {})
    assert after["published_at"][0].as_py() == datetime(
        2020, 1, 1, 4, 0, tzinfo=timezone.utc
    )
    assert after.schema.field("published_at").type == pa.timestamp("us", tz="UTC")
    assert len(notes) == 1
    row["published_at"] = datetime(2020, 1, 1, 5, 0)
    with pytest.raises(ValueError, match="differs from raw"):
        transform(pa.Table.from_pylist([row]), 0, 0, {})


@pytest.mark.parametrize("exclusions_only", [False, True])
def test_revision_preserves_parent_reindexes_metadata_and_invalidates_changed_cache(
    tmp_path, exclusions_only,
):
    parent, plan_dir, output = (tmp_path / n for n in ("parent", "plan", "output"))
    (parent / "records").mkdir(parents=True)
    (parent / "evidence").mkdir()
    revision = "approved-content-exclusions" if exclusions_only else REVISION
    if exclusions_only:
        (parent / "evidence" / REVISION).mkdir()
    plan_dir.mkdir()
    yt = [record("youtube-commons", "keep"), record("youtube-commons", "drop")]
    yt[0]["source_url"] = "youtube.com/watch?v=abcdefghijk"
    groups = [
        [record("arxiv", "drop"), record("arxiv", "keep")],
        yt,
        [nvd_record()],
        [record("test", "keep")],
    ]
    files, notes = [], []
    for i, group in enumerate(groups):
        relative = f"records/{i:06d}.parquet"
        path = parent / relative
        pq.write_table(pa.Table.from_pylist(group), path)
        files.append(
            dict(
                path=str(path),
                relative_path=relative,
                sha256=final.sha(path),
                bytes=path.stat().st_size,
                records=len(group),
                tokens=sum(r["content_length"] for r in group),
            )
        )
        notes.extend(
            dict(
                file=i,
                output_row_index=j,
                row_index=j,
                source=r["source_id"],
                record_id=r["record_id"],
                content_hash=r["content_hash"],
                license=r["license"],
                upstream_metadata={"preserve": "original"},
            )
            for j, r in enumerate(group)
        )
    with gzip.open(parent / "record-metadata.jsonl.gz", "wt") as f:
        for note in notes:
            f.write(json.dumps(note) + "\n")
    totals = {k: sum(e[k] for e in files) for k in ("records", "tokens", "bytes")}
    write(parent / "schema.json", {"test": "schema"})
    write(parent / "corpus-manifest.json", dict(files=files, totals=totals))
    write(
        parent / "final-prepared.json",
        dict(
            prepared=True,
            version=final.VERSION,
            manifest_sha256=final.sha(parent / "corpus-manifest.json"),
            record_metadata_sha256=final.sha(parent / "record-metadata.jsonl.gz"),
            schema_sha256=final.sha(parent / "schema.json"),
            evidence_sha256={},
            totals=totals,
            retained_text_and_quality_columns_unchanged=True,
        ),
    )
    assert final.audit(parent, workers=1)["integrity_passed"]
    old_hashes = {e["relative_path"]: e["sha256"] for e in files}
    excluded = [proof(groups[0][0], 0, 0), proof(groups[1][1], 1, 1)]
    write(
        plan_dir / "approved-plan.json",
        dict(
            approved=True,
            revision=revision,
            mode="exclusions-only" if exclusions_only else "metadata-repair",
            parent_manifest_sha256=final.sha(parent / "corpus-manifest.json"),
            exclusions=excluded,
            supporting_evidence_sha256={},
            expected=dict(
                excluded_records=2,
                excluded_tokens=sum(x["tokens"] for x in excluded),
                source_url=0 if exclusions_only else 1,
                published_at=0 if exclusions_only else 1,
            ),
        ),
    )
    result = prepare(parent, plan_dir / "approved-plan.json", output)
    assert result["changed_files"] == ([0, 1] if exclusions_only else [0, 1, 2])
    assert result["totals"]["records"] == 4
    for name, digest in old_hashes.items():
        assert final.sha(parent / name) == digest
    with sqlite3.connect(output / "quality/checkpoints.sqlite") as db:
        assert db.execute("select file from checked order by file").fetchall() == (
            [(2,), (3,)] if exclusions_only else [(3,)]
        )
    with gzip.open(output / "record-metadata.jsonl.gz", "rt") as f:
        metadata = [json.loads(line) for line in f]
    assert [x["output_row_index"] for x in metadata] == [0, 0, 0, 0]
    assert metadata[0]["previous_revision_output_row_index"] == 1
    assert metadata[1]["upstream_metadata"] == {"preserve": "original"}
    if exclusions_only:
        assert "canonical_metadata_revisions" not in metadata[1]
        assert pq.read_table(output / "records/000001.parquet").equals(
            pa.Table.from_pylist(yt).slice(0, 1), check_metadata=True
        )
        assert final.sha(output / "records/000002.parquet") == old_hashes[
            "records/000002.parquet"
        ]
    else:
        assert metadata[1]["canonical_metadata_revisions"][0]["fields"]["source_url"][
            "after"
        ].startswith("https://")
    assert final.audit(output, workers=1)["integrity_passed"]
    excluded_file = output / "evidence" / revision / "excluded-000000.parquet"
    assert pq.read_table(excluded_file)["record_id"].to_pylist() == ["arxiv:drop"]

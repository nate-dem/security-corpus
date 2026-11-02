from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.release.apply_upstream_license_labels import (
    apply,
    declarations,
    label,
    read,
    sha,
    write,
)

EVIDENCE = (
    Path(__file__).resolve().parents[1]
    / "tests/fixtures/upstream_licenses/sources.json"
)


def prepared(rule, old=False):
    values = dict(
        source=[rule["source"]],
        dataset_revision=[rule["dataset_revision"]],
        text=["unchanged security content"],
        content_length=[3],
        content_hash=["abc"],
    )
    if old:
        values["license"] = ["original label"]
    return pa.table(values)


@pytest.mark.parametrize("old", [False, True])
def test_labels_preserve_all_other_columns_and_original_values(old):
    rules = declarations(EVIDENCE)
    for rule in rules.values():
        before = prepared(rule, old)
        after, audit = label(before, rules)
        assert after["license"].to_pylist() == [rule["license"]]
        cols = [c for c in before.column_names if c != "license"]
        assert before.select(cols).equals(after.select(cols), check_metadata=True)
        assert audit[0]["old_field_present"] == old
        assert audit[0]["old_license"] == ("original label" if old else None)


def test_revision_mismatch_and_unmapped_source_rejected():
    rules = declarations(EVIDENCE)
    rule = next(iter(rules.values()))
    with pytest.raises(ValueError, match="revision"):
        label(prepared(dict(rule, dataset_revision="wrong")), rules)
    with pytest.raises(ValueError, match="Unmapped"):
        label(prepared(dict(rule, source="unknown")), rules)


def fixture_export(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    tables = [
        pa.table(
            dict(
                source_id=["arxiv"],
                license=["specific paper license"],
                content=["paper"],
            )
        )
    ]
    tables.extend(prepared(r) for r in declarations(EVIDENCE).values())
    entries = []
    for i, table in enumerate(tables):
        path = root / f"{i}.parquet"
        pq.write_table(table, path)
        entries.append(
            dict(path=str(path), sha256=sha(path), records=1, bytes=path.stat().st_size)
        )
    write(root / "corpus-manifest.json", dict(files=entries))
    (root / "license-changes.jsonl").write_text("")
    write(
        root / "corrections-complete.json",
        dict(
            complete=True,
            manifest_sha256=sha(root / "corpus-manifest.json"),
            license_changes_sha256=sha(root / "license-changes.jsonl"),
        ),
    )
    return root, entries


def test_full_export_preserves_baseline_and_original_inputs(tmp_path):
    root, entries = fixture_export(tmp_path)
    output = tmp_path / "export"
    result = apply(root, EVIDENCE, output)
    assert result["complete"] and not result["release_ready"]
    assert sum(result["labeled_records_by_source"].values()) == 3
    manifest = read(output / "corpus-manifest.json")
    assert manifest["files"][0] == entries[0]
    assert pq.read_table(entries[0]["path"])["license"].to_pylist() == [
        "specific paper license"
    ]
    for entry in entries:
        assert sha(entry["path"]) == entry["sha256"]
    assert (output / manifest["license_corrections"]).exists()
    with pytest.raises(ValueError, match="new directory"):
        apply(root, EVIDENCE, output)


def test_corrupt_input_rejected_before_output(tmp_path):
    root, entries = fixture_export(tmp_path)
    Path(entries[1]["path"]).write_bytes(b"corrupt")
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="checksum"):
        apply(root, EVIDENCE, output)
    assert not output.exists()

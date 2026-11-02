"""Label a completed local license-corrected export; never edit snapshot inputs.

Run after finish_license_copy_from_mac.sh. --input is its corrected draft directory;
--declarations is reports/release/upstream-licenses-20261021/sources.json;
--output must be a new directory. Unchanged payloads remain referenced by manifest.
The license column records upstream declarations, not permission determinations.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def declarations(path):
    result = {}
    for item in read(path)["sources"]:
        evidence = (path.parent / item["card_file"]).resolve()
        require(
            evidence.is_relative_to(path.parent.resolve()), "Evidence outside directory"
        )
        require(sha(evidence) == item["card_sha256"], "Evidence checksum mismatch")
        if evidence.suffix == ".json":
            card = read(evidence)
            require(
                card["sha"] == item["dataset_revision"], "Evidence revision mismatch"
            )
            require(
                card["cardData"]["license"] == "odc-by"
                and item["license"] == "ODC-By-1.0",
                "License declaration mismatch",
            )
        else:
            frontmatter = evidence.read_text().split("---", 2)[1]
            require(
                "license: cc-by-4.0" in frontmatter.splitlines()
                and item["license"] == "CC-BY-4.0",
                "License declaration mismatch",
            )
        require(item["source"] not in result, "Duplicate source declaration")
        result[item["source"]] = item
    return result


def label(table, rules):
    """Return a table and row-addressed old values; preserve all other columns."""
    if "source" not in table.column_names:
        require("license" in table.column_names, "Baseline missing license column")
        return table, []
    sources = table["source"].to_pylist()
    require(all(s in rules for s in sources), "Unmapped prepared source")
    revisions = table["dataset_revision"].to_pylist()
    existed = "license" in table.column_names
    old = table["license"].to_pylist() if existed else [None] * len(table)
    values, audit = [], []
    for row, (source, revision) in enumerate(zip(sources, revisions)):
        rule = rules[source]
        require(revision == rule["dataset_revision"], "Record revision mismatch")
        values.append(rule["license"])
        audit.append(
            dict(
                row=row,
                source=source,
                old_license=old[row],
                old_field_present=existed,
                new_license=rule["license"],
            )
        )
    column = pa.array(values, type=pa.string())
    after = (
        table.set_column(table.schema.get_field_index("license"), "license", column)
        if existed
        else table.append_column("license", column)
    )
    return after, audit


def apply(input_dir, declarations_path, output):
    input_dir, declarations_path, output = (
        p.resolve() for p in (input_dir, declarations_path, output)
    )
    require(not output.exists(), "Output must be a new directory")
    marker = read(input_dir / "corrections-complete.json")
    manifest_path = input_dir / "corpus-manifest.json"
    require(
        marker["complete"] and sha(manifest_path) == marker["manifest_sha256"],
        "Incomplete or changed corrected export",
    )
    require(
        sha(input_dir / "license-changes.jsonl") == marker["license_changes_sha256"],
        "Correction audit changed",
    )
    manifest = read(manifest_path)
    rules = declarations(declarations_path)
    for entry in manifest["files"]:
        source = Path(entry["path"])
        require(
            source.is_absolute() and not source.is_relative_to(output),
            "Invalid input path",
        )
        require(sha(source) == entry["sha256"], f"Input checksum mismatch: {source}")
    output.mkdir(parents=True)
    (output / "records").mkdir()
    (output / "evidence").mkdir()
    shutil.copyfile(declarations_path, output / "evidence/sources.json")
    for rule in rules.values():
        shutil.copyfile(
            declarations_path.parent / rule["card_file"],
            output / "evidence" / rule["card_file"],
        )
    shutil.copyfile(
        input_dir / "license-changes.jsonl", output / "prior-license-changes.jsonl"
    )
    files, counts = [], {}
    with (output / "upstream-license-changes.jsonl").open("w") as audit:
        for index, entry in enumerate(manifest["files"]):
            source = Path(entry["path"])
            require(sha(source) == entry["sha256"], "Input changed")
            before = pq.ParquetFile(source).read()
            after, changes = label(before, rules)
            require(len(before) == entry["records"], "Record count mismatch")
            updated = dict(entry)
            if changes:
                target = output / "records" / f"{index:06d}.parquet"
                pq.write_table(after, target, compression="zstd")
                checked = pq.ParquetFile(target).read()
                require(
                    after.equals(checked, check_metadata=True), "Roundtrip mismatch"
                )
                cols = [c for c in before.column_names if c != "license"]
                require(
                    before.select(cols).equals(
                        checked.select(cols), check_metadata=True
                    ),
                    "Non-license data changed",
                )
                updated.update(
                    path=str(target),
                    sha256=sha(target),
                    bytes=target.stat().st_size,
                    upstream_label_input_path=str(source),
                    upstream_label_input_sha256=entry["sha256"],
                )
                for change in changes:
                    counts[change["source"]] = counts.get(change["source"], 0) + 1
                    audit.write(
                        json.dumps(
                            dict(
                                change,
                                input_sha256=entry["sha256"],
                                input_path=str(source),
                                output_sha256=updated["sha256"],
                            )
                        )
                        + "\n"
                    )
            require(sha(source) == entry["sha256"], "Input changed during export")
            files.append(updated)
    require(sha(manifest_path) == marker["manifest_sha256"], "Input manifest changed")
    write(
        output / "license-field-definition.json",
        dict(
            meaning="Upstream-declared license; not a determination of redistribution permission.",
            evidence="evidence/sources.json",
            baseline="Existing record-specific labels retained; prior corrections preserved separately.",
        ),
    )
    write(
        output / "corpus-manifest.json",
        dict(
            manifest,
            files=files,
            license_corrections="prior-license-changes.jsonl",
            upstream_license_changes="upstream-license-changes.jsonl",
            upstream_label_parent_manifest=str(manifest_path),
            upstream_label_parent_sha256=marker["manifest_sha256"],
            release_ready=False,
        ),
    )
    result = dict(
        complete=True,
        release_ready=False,
        labeled_records_by_source=counts,
        manifest_sha256=sha(output / "corpus-manifest.json"),
        audit_sha256=sha(output / "upstream-license-changes.jsonl"),
        declarations_sha256=sha(output / "evidence/sources.json"),
        content_and_other_columns_unchanged=True,
    )
    write(output / "upstream-labels-complete.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "declarations", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(apply(args.input, args.declarations, args.output), indent=2))

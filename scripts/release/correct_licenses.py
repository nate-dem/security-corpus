"""Prepare verified license corrections; apply to a separate completed-snapshot copy.

No source files are changed. Only existing license values change; all other
columns, row order and token counts are checked for equality. This is metadata
repair, not a decision about which sources to publish or a publication approval.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def build_plan(ledger, sigma, cisa, output):
    ledger, sigma, cisa, output = map(Path, (ledger, sigma, cisa, output))
    ledger_digest = sha(ledger)
    with sqlite3.connect(ledger.resolve().as_uri() + "?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        files = {r["path"]: dict(r) for r in con.execute("SELECT * FROM files")}
        selected = {
            (r["input_path"], r["row_index"]): dict(r)
            for r in con.execute(
                "SELECT * FROM records WHERE source IN ('sigma','cisa-kev')"
            )
        }
    sigma_summary = read(sigma / "summary.json")
    require(sigma_summary["ledger_sha256"] == ledger_digest, "Sigma ledger differs")
    require(
        sigma_summary["records_sha256"] == sha(sigma / "records.jsonl"),
        "Sigma evidence changed",
    )
    require(
        sigma_summary["source_notice_sha256"] == sha(sigma / "source-LICENSE.txt"),
        "Sigma notice changed",
    )
    require(sigma_summary["source_license"] == "DRL-1.1", "Unexpected Sigma license")
    patches = defaultdict(list)
    covered = set()

    def add(row, new_license, evidence):
        key = (row["input_path"], row["row_index"])
        require(key not in covered, "Duplicate correction target")
        covered.add(key)
        patches[row["input_path"]].append(
            {
                "row_index": row["row_index"],
                "source": row["source"],
                "record_id": row["record_id"],
                "content_hash": row["content_hash"],
                "old_license": row["original_license"],
                "new_license": new_license,
                "evidence": evidence,
            }
        )

    with (sigma / "records.jsonl").open() as handle:
        for line in handle:
            proof = json.loads(line)
            row = selected[(proof["input_path"], proof["row_index"])]
            require(
                row["source"] == "sigma" and row["original_license"] == "LGPL-2.1",
                "Unexpected Sigma row",
            )
            require(
                proof["record_id"] == row["record_id"]
                and proof["content_hash"] == row["content_hash"],
                "Sigma identity mismatch",
            )
            require(
                proof["rule_source_matches"]
                and proof["content_hash_matches"]
                and proof["author"],
                "Sigma source verification incomplete",
            )
            require(
                proof["observed_source_license"] == "DRL-1.1"
                and proof["source_revision"] == sigma_summary["source_revision"],
                "Sigma terms mismatch",
            )
            add(
                row,
                "DRL-1.1",
                {
                    k: proof[k]
                    for k in (
                        "author",
                        "source_revision",
                        "source_url",
                        "notice_url",
                        "license_url",
                        "raw_file_sha256",
                    )
                },
            )

    source_inventory = read(cisa / "sources.json")
    source_files = {r["name"]: r for r in source_inventory["files"]}
    for name in ("LICENSE", "known_exploited_vulnerabilities.json"):
        item = source_files[name]
        require(sha(cisa / name) == item["sha256"], "CISA evidence changed")
        require(
            item["url"]
            == f"https://raw.githubusercontent.com/cisagov/kev-data/{source_inventory['revision']}/{name}",
            "Unexpected CISA provenance",
        )
    require(
        "The KEV database is distributed under the Creative Commons 0 1.0 License."
        in (cisa / "LICENSE").read_text(),
        "CISA license statement missing",
    )
    catalog = read(cisa / "known_exploited_vulnerabilities.json")
    upstream = {r["cveID"]: r for r in catalog["vulnerabilities"]}
    require(
        len(upstream) == len(catalog["vulnerabilities"]), "Duplicate CISA source IDs"
    )
    cisa_paths = {
        r["input_path"] for r in selected.values() if r["source"] == "cisa-kev"
    }
    for path in sorted(cisa_paths):
        require(sha(path) == files[path]["sha256"], "CISA payload changed")
        table = pq.ParquetFile(path).read()
        for index, payload in enumerate(table.to_pylist()):
            row = selected.get((path, index))
            if row is None or row["source"] != "cisa-kev":
                continue
            require(
                row["original_license"] == "CISA Terms of Use",
                "Unexpected CISA stored license",
            )
            original = upstream.get(payload["source_record_id"])
            require(
                original is not None,
                f"CISA record missing from pinned catalog: {row['record_id']}",
            )
            require(
                payload["content"] == original["shortDescription"],
                f"CISA text differs: {row['record_id']}",
            )
            require(
                hashlib.sha256(payload["content"].encode()).hexdigest()
                == row["content_hash"],
                "CISA content hash mismatch",
            )
            add(
                row,
                "CC0-1.0",
                {
                    "catalog_version": catalog["catalogVersion"],
                    "source_revision": source_inventory["revision"],
                    "source_url": source_files["known_exploited_vulnerabilities.json"][
                        "url"
                    ],
                    "notice_url": source_files["LICENSE"]["url"],
                    "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                    "catalog_sha256": source_files[
                        "known_exploited_vulnerabilities.json"
                    ]["sha256"],
                },
            )
    require(covered == set(selected), "Verified source coverage is incomplete")
    require(sha(ledger) == ledger_digest, "Ledger changed while reading")
    # Bind the exact original Parquet bytes for both sources before planning edits.
    for path in patches:
        require(sha(path) == files[path]["sha256"], f"Payload changed: {path}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "evidence").mkdir()
    evidence_files = {}
    for name, path in {
        "sigma-records.jsonl": sigma / "records.jsonl",
        "sigma-summary.json": sigma / "summary.json",
        "sigma-source-LICENSE.txt": sigma / "source-LICENSE.txt",
        "cisa-sources.json": cisa / "sources.json",
        "cisa-LICENSE.txt": cisa / "LICENSE",
        "cisa-catalog.json": cisa / "known_exploited_vulnerabilities.json",
    }.items():
        target = output / "evidence" / name
        shutil.copyfile(path, target)
        evidence_files[f"evidence/{name}"] = sha(target)
    plan = {
        "version": "verified-license-corrections-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "release_ready": False,
        "ledger_sha256": ledger_digest,
        "evidence_files": evidence_files,
        "files": [
            {
                "original_path": path,
                "input_sha256": files[path]["sha256"],
                "records": files[path]["records"],
                "patches": sorted(rows, key=lambda r: r["row_index"]),
            }
            for path, rows in sorted(patches.items())
        ],
        "corrections_by_source": dict(Counter(r["source"] for r in selected.values())),
        "deferred": "arXiv version binding; NVD/custom terms; assembled Q&A attribution; unresolved Reddit, logs, web and video rights. No new license inferred.",
    }
    write(output / "plan.json", plan)
    (output / "plan.sha256").write_text(sha(output / "plan.json") + "\n")
    return plan


def completed_manifest(snapshot):
    marker = read(snapshot / "checks-complete.json")
    require(
        marker.get("cpu_stages_complete") is True, "Snapshot CPU stages are incomplete"
    )
    for key, name in (
        ("summary_sha256", "summary.json"),
        ("rights_summary_sha256", "rights/summary.json"),
        ("integrity_summary_sha256", "integrity/summary.json"),
    ):
        require(sha(snapshot / name) == marker[key], f"Snapshot report changed: {name}")
    summary = read(snapshot / "summary.json")
    manifest_path = snapshot / "corpus-manifest.json"
    digest = sha(manifest_path)
    require(
        summary.get("complete") is True and summary["corpus_manifest_sha256"] == digest,
        "Snapshot manifest is not complete/bound",
    )
    for name in ("rights", "integrity"):
        report = read(snapshot / name / "summary.json")
        require(
            report.get("audit_complete") is True
            and report["manifest_sha256"] == digest,
            f"Unbound/incomplete {name} report",
        )
        if name == "rights":
            require(
                sha(snapshot / "rights/records.sqlite") == report["ledger_sha256"],
                "Snapshot rights ledger changed",
            )
    return read(manifest_path), digest


def local_path(remote, mappings):
    original = Path(remote)
    require(
        original.is_absolute() and ".." not in original.parts, "Unsafe manifest path"
    )
    matches = []
    for source, destination in mappings:
        source, destination = Path(source), Path(destination).resolve()
        require(source.is_absolute() and source != Path("/"), "Invalid path-map root")
        if original.is_relative_to(source):
            candidate = (destination / original.relative_to(source)).resolve()
            require(
                candidate.is_relative_to(destination), "Input escapes mapped directory"
            )
            matches.append(candidate)
    require(len(matches) == 1, f"Expected exactly one local mapping for {remote}")
    return matches[0]


def patched_table(table, patches):
    require("license" in table.schema.names, "No existing license column")
    licenses = table["license"].to_pylist()
    seen = set()
    for patch in patches:
        index = patch["row_index"]
        require(
            type(index) is int and 0 <= index < len(table) and index not in seen,
            "Invalid/duplicate patch row",
        )
        seen.add(index)
        for column, expected in (
            ("source_id", patch["source"]),
            ("record_id", patch["record_id"]),
            ("content_hash", patch["content_hash"]),
            ("license", patch["old_license"]),
        ):
            require(
                table[column][index].as_py() == expected,
                f"Patch identity/old value differs at row {index}: {column}",
            )
        licenses[index] = patch["new_license"]
    field = table.schema.field("license")
    return table.set_column(
        table.schema.get_field_index("license"),
        field,
        pa.array(licenses, type=field.type),
    )


def apply_plan(plan_path, snapshot, mappings, output):
    plan_path, snapshot, output = map(Path, (plan_path, snapshot, output))
    output = output.resolve()
    require(
        sha(plan_path) == plan_path.with_suffix(".sha256").read_text().strip(),
        "Correction plan changed since preparation",
    )
    plan = read(plan_path)
    require(
        plan["version"] == "verified-license-corrections-v1",
        "Unsupported correction plan",
    )
    for name, digest in plan["evidence_files"].items():
        evidence = (plan_path.parent / name).resolve()
        require(
            evidence.is_relative_to(plan_path.parent.resolve())
            and sha(evidence) == digest,
            "Correction evidence changed",
        )
    manifest, manifest_digest = completed_manifest(snapshot)
    for _, root in mappings:
        root = Path(root).resolve()
        require(
            not output.is_relative_to(root) and not root.is_relative_to(output),
            "Keep corrected output separate from backup roots",
        )
    input_files = []
    for entry in manifest["files"]:
        path = local_path(entry["path"], mappings)
        require(
            path != output and not path.is_relative_to(output), "Output overlaps input"
        )
        require(sha(path) == entry["sha256"], f"Local copy checksum mismatch: {path}")
        input_files.append(
            {**entry, "path": str(path), "original_remote_path": entry["path"]}
        )
    require(
        len({f["path"] for f in input_files}) == len(input_files), "Repeated input path"
    )
    by_digest = defaultdict(list)
    for entry in input_files:
        by_digest[entry["sha256"]].append(entry)
    planned = set()
    for change in plan["files"]:
        digest = change["input_sha256"]
        require(
            digest not in planned and len(by_digest[digest]) == 1,
            "Correction file missing or ambiguous in final snapshot",
        )
        require(
            by_digest[digest][0]["records"] == change["records"],
            "Correction file row count differs",
        )
        planned.add(digest)
        table = pq.ParquetFile(by_digest[digest][0]["path"]).read()
        require(len(table) == change["records"], "Parquet row count differs")
        patched_table(table, change["patches"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "records").mkdir()
    shutil.copytree(plan_path.parent / "evidence", output / "evidence")
    shutil.copyfile(plan_path, output / "correction-plan.json")
    updated = {}
    audit_path = output / "license-changes.jsonl"
    with audit_path.open("w") as audit:
        for change in plan["files"]:
            entry = by_digest[change["input_sha256"]][0]
            source = Path(entry["path"])
            require(sha(source) == entry["sha256"], "Input changed before patch")
            before = pq.ParquetFile(source).read()
            after = patched_table(before, change["patches"])
            target = output / "records" / f"{entry['sha256']}.parquet"
            pq.write_table(after, target, compression="zstd")
            checked = pq.ParquetFile(target).read()
            require(
                after.equals(checked, check_metadata=True),
                "Parquet round-trip mismatch",
            )
            columns = [c for c in before.schema.names if c != "license"]
            require(
                before.select(columns).equals(
                    checked.select(columns), check_metadata=True
                ),
                "Non-license values changed",
            )
            require(sha(source) == entry["sha256"], "Original input changed")
            updated[entry["sha256"]] = {
                **entry,
                "path": str(target),
                "sha256": sha(target),
                "bytes": target.stat().st_size,
                "original_sha256": entry["sha256"],
                "original_local_path": str(source),
            }
            for patch in change["patches"]:
                audit.write(
                    json.dumps(
                        {
                            **patch,
                            "original_file_sha256": entry["sha256"],
                            "corrected_file_sha256": updated[entry["sha256"]]["sha256"],
                        }
                    )
                    + "\n"
                )
    require(
        sha(snapshot / "corpus-manifest.json") == manifest_digest,
        "Snapshot manifest changed",
    )
    write(
        output / "corpus-manifest.json",
        {
            **manifest,
            "files": [updated.get(e["sha256"], e) for e in input_files],
            "original_manifest_sha256": manifest_digest,
            "license_corrections": "license-changes.jsonl",
            "release_ready": False,
            "scope": "Draft local manifest with verified license corrections. Unchanged files reference the local backup. Existing rights/integrity reports describe the original snapshot; publication review remains required.",
        },
    )
    result = {
        "complete": True,
        "release_ready": False,
        "corrections_by_source": plan["corrections_by_source"],
        "changed_files": len(updated),
        "content_and_other_columns_unchanged": True,
        "original_snapshot_manifest_sha256": manifest_digest,
        "correction_plan_sha256": sha(plan_path),
        "license_changes_sha256": sha(audit_path),
        "manifest_sha256": sha(output / "corpus-manifest.json"),
    }
    write(output / "corrections-complete.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("build-plan")
    for name in ("ledger", "sigma", "cisa", "output"):
        build.add_argument(f"--{name}", type=Path, required=True)
    apply = sub.add_parser("apply")
    for name in ("plan", "snapshot", "output"):
        apply.add_argument(f"--{name}", type=Path, required=True)
    apply.add_argument(
        "--path-map",
        nargs=2,
        action="append",
        required=True,
        metavar=("REMOTE_ROOT", "LOCAL_ROOT"),
    )
    args = parser.parse_args()
    if args.action == "build-plan":
        result = build_plan(args.ledger, args.sigma, args.cisa, args.output)
    else:
        result = apply_plan(args.plan, args.snapshot, args.path_map, args.output)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"files", "evidence_files"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

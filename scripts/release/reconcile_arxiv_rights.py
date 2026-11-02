"""Supplement an existing rights ledger with original arXiv license evidence.

This reads metadata, preserves all observations, and never approves publication
or changes corpus rows. OAI metadata alone does not bind a license to the exact
downloaded paper version. Run separately from the frozen snapshot CPU job.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from urllib.parse import urlparse

from ingest.connectors.arxiv.metadata import _parse_oai_record
import ingest.connectors.arxiv.metadata as parser_module


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_license(value):
    """Normalize spelling only; never infer CC0 from a public-domain label."""
    if not value:
        return None
    value = value.strip()
    if value == "Public Domain":
        return "UNSPECIFIED_PUBLIC_DOMAIN"
    if value == "arXiv Perpetual Non-Exclusive License":
        return "arXiv-nonexclusive-distrib-1.0"
    if value == "CC0-1.0" or re.fullmatch(
        r"CC-BY(?:-NC)?(?:-SA|-ND)?-(?:2\.5|3\.0|4\.0)", value
    ):
        return value
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        return "UNRECOGNIZED:" + value
    path = parsed.path.rstrip("/")
    if parsed.hostname in {"creativecommons.org", "www.creativecommons.org"}:
        if path == "/publicdomain/zero/1.0":
            return "CC0-1.0"
        if path == "/licenses/publicdomain":
            return "Legacy-CC-Public-Domain-Dedication"
        match = re.fullmatch(
            r"/licenses/(by(?:-nc)?(?:-sa|-nd)?)/(2\.5|3\.0|4\.0)", path
        )
        if match:
            return f"CC-{match[1].upper()}-{match[2]}"
    if parsed.hostname in {"arxiv.org", "www.arxiv.org"} and path in {
        "/licenses/nonexclusive-distrib/1.0",
        "/licenses/nonexclusive-distrib/1.0/license.html",
    }:
        return "arXiv-nonexclusive-distrib-1.0"
    return "UNRECOGNIZED:" + value


def reconcile(stored, observations):
    """Describe agreement and uncertainty, without selecting a favorable grant."""
    if not observations:
        return "missing_raw_metadata", None
    licenses = {o["canonical_license"] for o in observations if o["license_url"]}
    if len(licenses) > 1:
        return "multiple_observed_licenses", None
    if any(not o["license_url"] for o in observations):
        return "missing_explicit_raw_license", None
    observed = next(iter(licenses))
    original = canonical_license(stored)
    if observed.startswith("UNRECOGNIZED:"):
        return "unrecognized_license_needs_review", observed
    if original == "UNSPECIFIED_PUBLIC_DOMAIN" and observed == "CC0-1.0":
        return "public_domain_label_resolved_to_cc0", observed
    if original != observed:
        return "stored_raw_license_conflict", observed
    if observed == "Legacy-CC-Public-Domain-Dedication":
        return "legacy_public_domain_needs_review", observed
    return "license_label_matches", observed


def run(ledger, metadata_dirs, output):
    ledger = Path(ledger).resolve()
    output = Path(output).resolve()
    ledger_digest = sha256(ledger)
    with sqlite3.connect(ledger.as_uri() + "?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        records = [dict(r) for r in con.execute(
            "SELECT input_path,row_index,record_id,content_hash,tokens,"
            "original_license,source_url FROM records WHERE source='arxiv' "
            "ORDER BY input_path,row_index"
        )]
    if not records:
        raise ValueError("No arXiv rows in the supplied ledger")
    wanted = {r["record_id"].removeprefix("arxiv:") for r in records}
    observations = defaultdict(list)
    files = {}
    paths = sorted({
        p.resolve() for d in metadata_dirs for p in Path(d).glob("*.jsonl")
    })
    if not paths:
        raise ValueError("No raw metadata JSONL files found")
    for path in paths:
        digest = sha256(path)
        matches = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                # Fail on malformed JSON instead of silently losing evidence.
                raw = json.loads(line)
                parsed = _parse_oai_record(raw)
                if parsed is None or parsed["arxiv_id"] not in wanted:
                    continue
                observations[parsed["arxiv_id"]].append({
                    "metadata_file": str(path),
                    "line_number": line_number,
                    "datestamp": parsed["datestamp"],
                    "license_url": parsed["license_url"],
                    "canonical_license": canonical_license(parsed["license_url"]),
                })
                matches += 1
        if sha256(path) != digest:
            raise ValueError(f"Metadata changed while reading: {path}")
        files[str(path)] = {"sha256": digest, "selected_observations": matches}
    if sha256(ledger) != ledger_digest:
        raise ValueError("Rights ledger changed while reading")
    # Existing results are immutable; an interrupted run needs a new directory.
    output.mkdir(parents=True, exist_ok=False)
    statuses = defaultdict(Counter)
    labels = defaultdict(Counter)
    result_path = output / "records.jsonl"
    with result_path.open("w", encoding="utf-8") as handle:
        for record in records:
            observed = observations[record["record_id"].removeprefix("arxiv:")]
            status, candidate = reconcile(record["original_license"], observed)
            for counts in (statuses[status], labels[candidate or "unresolved"]):
                counts["records"] += 1
                counts["tokens"] += record["tokens"]
            record.update({
                "observations": observed,
                "evidence_status": status,
                "observed_license_candidate": candidate,
                "paper_version_binding_established": False,
                "public_release_approved": False,
            })
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "release_ready": False,
        "scope": "Metadata evidence comparison, not publication clearance. "
        "No downloaded paper version was bound to its license by this tool.",
        "ledger": {"path": str(ledger), "sha256": ledger_digest},
        "code": {str(p): sha256(p) for p in (
            Path(__file__).resolve(), Path(parser_module.__file__).resolve()
        )},
        "metadata_files": files,
        "records": len(records),
        "tokens": sum(r["tokens"] for r in records),
        "by_evidence_status": dict(sorted(statuses.items())),
        "by_observed_license_candidate": dict(sorted(labels.items())),
        "output": {"path": str(result_path), "sha256": sha256(result_path)},
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--metadata-dir", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = run(args.ledger, args.metadata_dir, args.output)
    print(json.dumps({k: summary[k] for k in (
        "records", "tokens", "by_evidence_status", "by_observed_license_candidate"
    )}, indent=2))


if __name__ == "__main__":
    main()

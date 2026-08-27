#!/usr/bin/env python3
"""Audit Parquet release inputs against the source-license release gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fnmatch import fnmatchcase
import json
from pathlib import Path
from typing import Any, Sequence

import duckdb
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = ROOT / "config" / "source_licenses.yaml"
BLOCKING_STATES = {"blocked", "review_required", "unknown"}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    files = _discover_parquet(args.inputs)
    if not files:
        raise FileNotFoundError("No Parquet files found in the requested inputs")
    policy = yaml.safe_load(args.policy.read_text(encoding="utf-8"))
    rows = _summarize(files)
    report = _build_report(rows, policy, files)

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 2 if report["release_blocking_records"] else 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def _discover_parquet(inputs: Sequence[Path]) -> list[Path]:
    files: set[Path] = set()
    for value in inputs:
        path = value.resolve()
        if path.is_file() and path.suffix == ".parquet":
            files.add(path)
        elif path.is_dir():
            files.update(candidate for candidate in path.rglob("*.parquet"))
    return sorted(files)


def _summarize(files: Sequence[Path]) -> list[tuple[str, str, int, int]]:
    paths = ", ".join(_sql_string(path.as_posix()) for path in files)
    connection = duckdb.connect()
    relation = f"read_parquet([{paths}], union_by_name = true, hive_partitioning = false)"
    columns = {
        row[0] for row in connection.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
    }
    required = {"source_id", "license", "content_length"}
    missing = sorted(required - columns)
    if missing:
        raise ValueError("Release input is missing columns: " + ", ".join(missing))
    rows = connection.execute(
        f"""
        SELECT
            coalesce(source_id, ''),
            coalesce(license, ''),
            count(*),
            coalesce(sum(content_length), 0)
        FROM {relation}
        GROUP BY source_id, license
        ORDER BY source_id, license
        """
    ).fetchall()
    connection.close()
    return [(str(a), str(b), int(c), int(d)) for a, b, c, d in rows]


def _build_report(
    rows: Sequence[tuple[str, str, int, int]],
    policy: dict[str, Any],
    files: Sequence[Path],
) -> dict[str, Any]:
    findings = []
    state_totals: dict[str, dict[str, int]] = {}
    for source_id, license_value, records, tokens in rows:
        source_policy = _match_policy(source_id, policy.get("policies", []))
        canonical_license = policy.get("license_aliases", {}).get(
            license_value,
            license_value,
        )
        state, issue = _classify_license(source_policy, canonical_license)
        totals = state_totals.setdefault(state, {"records": 0, "tokens": 0})
        totals["records"] += records
        totals["tokens"] += tokens
        findings.append(
            {
                "source_id": source_id,
                "license": license_value,
                "canonical_license": canonical_license,
                "state": state,
                "records": records,
                "tokens": tokens,
                "issue": issue,
            }
        )

    blocking_records = sum(
        values["records"]
        for state, values in state_totals.items()
        if state in BLOCKING_STATES
    )
    blocking_tokens = sum(
        values["tokens"]
        for state, values in state_totals.items()
        if state in BLOCKING_STATES
    )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy_schema_version": policy.get("schema_version"),
        "input_files": [path.as_posix() for path in files],
        "state_totals": state_totals,
        "release_blocking_records": blocking_records,
        "release_blocking_tokens": blocking_tokens,
        "findings": findings,
    }


def _match_policy(source_id: str, policies: Sequence[dict]) -> dict | None:
    for policy in policies:
        if any(fnmatchcase(source_id, pattern) for pattern in policy["source_patterns"]):
            return policy
    return None


def _classify_license(policy: dict | None, license_value: str) -> tuple[str, str | None]:
    if policy is None:
        return "unknown", "No source policy matches this source_id"
    expected = set(policy.get("expected_licenses", []))
    if expected and license_value not in expected:
        return "unknown", "License value does not match the reviewed source policy"

    state = policy["state"]
    if state == "per_license":
        state = policy.get("license_states", {}).get(license_value, "unknown")
    issue = policy.get("blocker") if state in BLOCKING_STATES else None
    return state, issue


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


if __name__ == "__main__":
    raise SystemExit(main())

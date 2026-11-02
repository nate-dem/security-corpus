"""Resumable CPU audit of a frozen set of finished YouTube partitions.

Uses the original raw-response verifier without changing its files or policy.
Each cached result is bound to all part files and the prepared input bytes.
No model is loaded, and worker outputs are never rewritten.
"""

import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import shutil
import time

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.curation import production as old
from scripts.curation.production_audit import read_rows
from scripts.curation_single import audit, run
from scripts.youtube.download import _directory_lock, _write_json
from scripts.youtube.profile import _sha256

HERE = Path(__file__).resolve().parent
LOG = logging.getLogger(__name__)
SOURCE = "youtube-commons"
COUNT_FIELDS = (
    "source",
    "records",
    "input_tokens",
    "candidate_records",
    "candidate_tokens",
    "unresolved_documents",
    "routes",
    "spans",
    "token_routes",
)


def read(path):
    return json.loads(Path(path).read_text())


def auditor_code():
    return {p.name: _sha256(p) for p in sorted(HERE.glob("*.py"))}


def source_slots(plan):
    return [
        i
        for i, part in enumerate(plan["parts"])
        if part["path"].split("/")[1] == SOURCE
    ]


def freeze(path, plan, output):
    binding = {
        "version": "resumable-youtube-audit-v1",
        "plan_sha256": _sha256(path),
        "plan_path": str(path),
        "output": str(output),
        "auditor_code": auditor_code(),
        "runtime": {
            "python": platform.python_version(),
            "pyarrow": pa.__version__,
            "tiktoken": importlib.metadata.version("tiktoken"),
        },
        "draws_per_route": 32,  # Existing audit sample size; not a filter.
        "source": SOURCE,
    }
    target = output / "frozen.json"
    if target.exists():
        saved = read(target)
        if any(saved.get(k) != v for k, v in binding.items()):
            raise ValueError("Frozen audit binding changed; preserve this run")
        slots = [entry["slot"] for entry in saved["parts"]]
        if (
            not slots
            or len(set(slots)) != len(slots)
            or not set(slots) <= set(source_slots(plan))
        ):
            raise ValueError("Invalid frozen audit slots")
        return saved
    parts = []
    for slot in source_slots(plan):
        summary = run.folder(plan, slot) / "summary.json"
        if not summary.exists():
            continue
        value = read(summary)
        if value.get("workflow_finished"):
            if (
                value.get("plan_sha256") != binding["plan_sha256"]
                or value.get("slot") != slot
                or value.get("source") != SOURCE
            ):
                raise ValueError(f"Finished slot {slot} has incompatible provenance")
            parts.append({"slot": slot, "summary_sha256": _sha256(summary)})
    if not parts:
        raise ValueError("No completed YouTube partitions to audit")
    binding.update(parts=parts, planned_youtube_parts=len(source_slots(plan)))
    _write_json(target, binding)
    return binding


def fingerprint(plan, entry):
    """Bind actual bytes, including raw responses, rather than timestamps alone."""
    slot = entry["slot"]
    directory = run.folder(plan, slot)
    if _sha256(directory / "summary.json") != entry["summary_sha256"]:
        raise ValueError(f"Frozen worker summary changed: slot {slot}")
    part = plan["parts"][slot]
    source = Path(plan["input_dir"]) / part["path"]
    if _sha256(source) != part["sha256"]:
        raise ValueError(f"Prepared input changed: slot {slot}")
    # Worker files are immutable once complete. Exclude only the advisory lock;
    # extra or changed response/checkpoint files also invalidate cached audits.
    files = sorted(
        p for p in directory.rglob("*") if p.is_file() and p.name != ".download.lock"
    )
    result = hashlib.sha256(part["sha256"].encode())
    for file in files:
        result.update(str(file.relative_to(directory)).encode() + b"\0")
        result.update(_sha256(file).encode() + b"\n")
    return result.hexdigest()


@contextmanager
def bound_plan_digest(path, digest):
    """Avoid rehashing the large plan for each document in the legacy verifier.

    This single-process audit temporarily supplies the already verified plan hash
    only for that exact path. Every other file is hashed normally. Verify the plan
    before and after, and restore the module binding even if verification fails.
    No source file, frozen plan, parser or filtering decision changes.
    """
    original = run._sha256
    if _sha256(path) != digest:
        raise ValueError("Plan changed before partition audit")

    def cached(target):
        return digest if Path(target) == path else original(target)

    run._sha256 = cached
    try:
        yield
    finally:
        run._sha256 = original
        if _sha256(path) != digest:
            raise ValueError("Plan changed during partition audit")


def verify_part(path, plan, output, frozen, entry):
    slot = entry["slot"]
    cache = output / "parts" / f"{slot:05d}"
    cache.mkdir(parents=True, exist_ok=True)
    marker, decisions = cache / "audit.json", cache / "decisions.parquet"
    started = time.monotonic()
    with _directory_lock(run.folder(plan, slot)):
        before = fingerprint(plan, entry)
        if marker.exists():
            report = read(marker)
            worker = read(run.folder(plan, slot) / "summary.json")
            if (
                report["input_fingerprint"] != before
                or report["snapshot_sha256"] != _sha256(output / "frozen.json")
                or report["plan_sha256"] != frozen["plan_sha256"]
                or report["auditor_code"] != frozen["auditor_code"]
                or report["summary_sha256"] != entry["summary_sha256"]
                or report["slot"] != slot
                or report["decisions_sha256"] != _sha256(decisions)
                or any(report[k] != worker[k] for k in COUNT_FIELDS)
            ):
                raise ValueError(f"Cached audit inputs or outputs changed: slot {slot}")
            return report, True
        with bound_plan_digest(path, frozen["plan_sha256"]):
            summary = run.checked_summary(path, plan, slot)
            counts, exports = run.audit_part(path, plan, slot)
        if (
            not summary
            or not summary.get("workflow_finished")
            or any(summary[k] != v for k, v in counts.items())
        ):
            raise ValueError(f"Worker summary differs from fresh audit: slot {slot}")
        if fingerprint(plan, entry) != before:
            raise ValueError(f"Part changed during verification: slot {slot}")
        temporary = decisions.with_suffix(".tmp")
        pq.write_table(
            pa.Table.from_pylist(exports["decisions.parquet"], schema=run.DECISIONS),
            temporary,
            compression="zstd",
        )
        temporary.replace(decisions)
        report = {
            "slot": slot,
            "source": summary["source"],
            **counts,
            "summary_sha256": entry["summary_sha256"],
            "plan_sha256": frozen["plan_sha256"],
            "snapshot_sha256": _sha256(output / "frozen.json"),
            "auditor_code": frozen["auditor_code"],
            "input_fingerprint": before,
            "decisions_sha256": _sha256(decisions),
            "audit_seconds": time.monotonic() - started,
        }
        _write_json(marker, report)
        return report, False


def finish(path, plan, output, frozen, reports):
    review = output / "review"
    frames = defaultdict(list)
    unique, unique_tokens = set(), 0
    locations = {}
    for report in reports:
        slot = report["slot"]
        for row in read_rows(output / "parts" / f"{slot:05d}" / "decisions.parquet"):
            if row["case_id"] in locations:
                raise ValueError("Repeated case identity across audited parts")
            locations[row["case_id"]] = slot
            frames[row["route"]].append(
                {"case_id": row["case_id"], "content_length": row["content_length"]}
            )
            if row["route"] == "eligible" and row["content_hash"] not in unique:
                unique.add(row["content_hash"])
                unique_tokens += row["content_length"]
    draws, strata, selected = [], [], set()
    for route, rows in sorted(frames.items()):
        sampled = audit.token_draws(
            rows,
            frozen["draws_per_route"],
            old.digest({"snapshot": _sha256(output / "frozen.json"), "route": route}),
        )
        draws.extend({"source": SOURCE, "route": route, **d} for d in sampled)
        selected.update(d["case_id"] for d in sampled)
        strata.append(
            {
                "source": SOURCE,
                "route": route,
                "documents": len(rows),
                "tokens": sum(r["content_length"] for r in rows),
                "draws": len(sampled),
            }
        )
    # Keep aggregation narrow; original texts are read only for selected parts.
    del frames
    wanted_slots = {locations[cid] for cid in selected}
    del locations
    cases, keys, spans = [], [], []
    for slot in sorted(wanted_slots):
        directory = run.folder(plan, slot)
        wanted = {
            row["case_id"]: row
            for row in read_rows(output / "parts" / f"{slot:05d}" / "decisions.parquet")
            if row["case_id"] in selected
        }
        keys.extend(wanted.values())
        hashes = {r["content_hash"] for r in wanted.values()}
        for row in read_rows(directory / "documents.parquet"):
            cid = old.case_id(row)
            if cid in wanted:
                if any(
                    row[k] != wanted[cid][k]
                    for k in ("source", "content_hash", "content_length")
                ):
                    raise ValueError("Sample differs from audited decision")
                cases.append({"case_id": cid, **row})
        spans.extend(
            {"slot": slot, **row}
            for row in read_rows(directory / "spans.parquet")
            if row["content_hash"] in hashes
        )
    if (
        {c["case_id"] for c in cases} != selected
        or len(cases) != len(selected)
        or len(keys) != len(selected)
    ):
        raise ValueError("Incomplete sampled text/decision coverage")
    _write_json(
        review / "review-packet.json",
        {
            "cases": cases,
            "scope": "Original completed YouTube texts; no source rewriting or release selection.",
        },
    )
    _write_json(
        review / "draws.json",
        {
            "strata": strata,
            "draws": draws,
            "scope": "32 uniform token draws with replacement per route. Preserve repeats. Inference applies only to this frozen completed-work snapshot.",
        },
    )
    _write_json(review / "audit-key.json", keys)
    _write_json(review / "sample-spans.json", spans)
    _write_json(review / "part-audits.json", reports)
    shutil.copyfile(output / "frozen.json", review / "frozen.json")
    shutil.copyfile(path, review / "plan.json")
    missing = sorted(set(source_slots(plan)) - {r["slot"] for r in reports})
    counts = {
        k: sum(r[k] for r in reports)
        for k in (
            "records",
            "input_tokens",
            "candidate_records",
            "candidate_tokens",
            "unresolved_documents",
        )
    }
    counts["parts"] = len(reports)
    names = (
        "review-packet.json",
        "draws.json",
        "audit-key.json",
        "sample-spans.json",
        "part-audits.json",
        "frozen.json",
        "plan.json",
    )
    return {
        "audit_complete": True,
        "all_planned_work_complete": not missing and not counts["unresolved_documents"],
        "planned_scope": "YouTube only; deliberately excludes CPU-filtered web sources",
        "plan_sha256": frozen["plan_sha256"],
        "completed_parts": len(reports),
        "planned_parts": len(source_slots(plan)),
        "unprocessed_slots": missing,
        "errors": [],
        "by_source": {SOURCE: counts},
        "candidate_exact_unique_tokens_in_this_snapshot": unique_tokens,
        "files": {name: _sha256(review / name) for name in names},
        "quality_review_complete": False,
        "release_ready": False,
        "scope": "Frozen finished-part snapshot. Source/export/raw-response integrity verified; checkpoint reuse requires unchanged input bytes. Candidates precede semantic review, cross-source deduplication and publication checks. No inference performed.",
    }


def execute(path, output):
    path, output = Path(path).resolve(), Path(output).resolve()
    plan = run.load_plan(path)
    for protected in (
        path.parent,
        Path(plan["input_dir"]).resolve(),
        HERE.parent.parent,
    ):
        if (
            output == protected
            or output.is_relative_to(protected)
            or protected.is_relative_to(output)
        ):
            raise ValueError(
                "Audit output must be separate from source, scoring and code"
            )
    output.mkdir(parents=True, exist_ok=True)
    with _directory_lock(output):
        review = output / "review"
        review.mkdir(exist_ok=True)
        frozen = freeze(path, plan, output)
        reports = []
        try:
            _write_json(
                review / "summary.json",
                {"audit_complete": False, "plan_sha256": frozen["plan_sha256"]},
            )
            LOG.info(
                "Frozen %s completed YouTube parts; verifying source lineage",
                len(frozen["parts"]),
            )
            _write_json(
                output / "progress.json",
                {
                    "stage": "verify_lineage",
                    "audit_complete": False,
                    "snapshot_parts": len(frozen["parts"]),
                },
            )
            if (
                _sha256(Path(plan["input_dir"]) / "lineage.parquet")
                != plan["lineage_sha256"]
            ):
                raise ValueError("Source alias lineage changed")
            for index, entry in enumerate(frozen["parts"]):
                state = {
                    "stage": "verify_parts",
                    "audit_complete": False,
                    "completed_parts": index,
                    "snapshot_parts": len(frozen["parts"]),
                    "current_slot": entry["slot"],
                }
                _write_json(output / "progress.json", state)
                LOG.info(
                    "[%s/%s] checking slot %s",
                    index + 1,
                    len(frozen["parts"]),
                    entry["slot"],
                )
                report, cached = verify_part(path, plan, output, frozen, entry)
                reports.append(report)
                LOG.info(
                    "[%s/%s] slot %s verified (%s); %s candidate tokens",
                    index + 1,
                    len(frozen["parts"]),
                    entry["slot"],
                    "cached" if cached else "fresh",
                    report["candidate_tokens"],
                )
            _write_json(
                output / "progress.json",
                {
                    "stage": "sample_and_export",
                    "audit_complete": False,
                    "completed_parts": len(reports),
                    "snapshot_parts": len(reports),
                },
            )
            result = finish(path, plan, output, frozen, reports)
            # A retained checkpoint is trusted only while all bound bytes agree.
            LOG.info("Final verification of all snapshot inputs")
            _write_json(
                output / "progress.json",
                {
                    "stage": "final_verification",
                    "audit_complete": False,
                    "completed_parts": len(reports),
                    "snapshot_parts": len(reports),
                },
            )
            for entry, report in zip(frozen["parts"], reports):
                if fingerprint(plan, entry) != report["input_fingerprint"]:
                    raise ValueError("Snapshot input changed before completion")
            if (
                _sha256(path) != frozen["plan_sha256"]
                or _sha256(Path(plan["input_dir"]) / "lineage.parquet")
                != plan["lineage_sha256"]
            ):
                raise ValueError("Plan or lineage changed during audit")
            _write_json(review / "summary.json", result)
            _write_json(
                output / "progress.json",
                {
                    "stage": "complete",
                    "audit_complete": True,
                    "completed_parts": len(reports),
                    "snapshot_parts": len(reports),
                    "candidate_tokens": result["by_source"][SOURCE]["candidate_tokens"],
                },
            )
            LOG.info(
                "AUDIT COMPLETE: %s parts; %s candidate tokens; reports %s",
                len(reports),
                result["by_source"][SOURCE]["candidate_tokens"],
                review,
            )
            return result
        except Exception as exc:
            failure = {
                "audit_complete": False,
                "plan_sha256": frozen["plan_sha256"],
                "verified_parts_this_invocation": len(reports),
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
            _write_json(review / "summary.json", failure)
            _write_json(output / "progress.json", {"stage": "failed", **failure})
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    execute(args.plan, args.output)


if __name__ == "__main__":
    main()

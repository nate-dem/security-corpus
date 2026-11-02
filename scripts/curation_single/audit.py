"""CPU revalidation and token-proportional review of completed single-pass work."""

import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path
import random
import tarfile

from scripts.curation import production as old
from scripts.curation.production_audit import read_rows
from scripts.youtube.download import _directory_lock, _write_json
from scripts.youtube.profile import _sha256
from . import run


def token_draws(rows, count, seed):
    """Uniform token positions with replacement, not uniform document selection."""
    rows = sorted(rows, key=lambda r: r["case_id"])
    ends, total = [], 0
    for row in rows:
        if row["content_length"] <= 0:
            raise ValueError("Invalid token weight")
        total += row["content_length"]
        ends.append(total)
    generator = random.Random(seed)
    draws = []
    for draw in range(count if total else 0):
        position = generator.randrange(total)
        index = bisect_right(ends, position)
        draws.append(
            {
                "draw": draw,
                "case_id": rows[index]["case_id"],
                "token_offset": position - (ends[index - 1] if index else 0),
                "population_tokens": total,
            }
        )
    return draws


def finalize(path, draws=32):
    plan = run.load_plan(path)
    output = Path(plan["output_dir"]) / "review"
    output.mkdir(exist_ok=True)
    plan_sha = _sha256(path)
    with _directory_lock(output):
        _write_json(
            output / "summary.json", {"audit_complete": False, "plan_sha256": plan_sha}
        )
        if (
            _sha256(Path(plan["input_dir"]) / "lineage.parquet")
            != plan["lineage_sha256"]
        ):
            raise ValueError("Source alias lineage changed")
        reports, errors, missing, frames = [], [], [], defaultdict(list)
        unique, unique_tokens = set(), 0
        for slot in range(len(plan["parts"])):
            try:
                summary = run.checked_summary(path, plan, slot)
                if not summary or not summary.get("workflow_finished"):
                    missing.append(slot)
                    continue
                counts, exports = run.audit_part(path, plan, slot)
                if any(summary[k] != v for k, v in counts.items()):
                    raise ValueError("Worker summary differs from fresh audit")
                reports.append(
                    {
                        "slot": slot,
                        "source": summary["source"],
                        **counts,
                        "generation_this_invocation": summary[
                            "generation_this_invocation"
                        ],
                        "wall_seconds_this_invocation": summary[
                            "wall_seconds_this_invocation"
                        ],
                        "summary_sha256": _sha256(
                            run.folder(plan, slot) / "summary.json"
                        ),
                    }
                )
                for row in exports["decisions.parquet"]:
                    frames[(row["source"], row["route"])].append({**row, "slot": slot})
                    if row["route"] == "eligible" and row["content_hash"] not in unique:
                        unique.add(row["content_hash"])
                        unique_tokens += row["content_length"]
            except (OSError, ValueError, KeyError) as error:
                errors.append({"slot": slot, "error": str(error)})
        selections, selected, strata = [], set(), []
        for (source, route), rows in sorted(frames.items()):
            sampled = token_draws(
                rows,
                draws,
                old.digest({"plan": plan_sha, "source": source, "route": route}),
            )
            selections.extend({"source": source, "route": route, **d} for d in sampled)
            selected.update(d["case_id"] for d in sampled)
            strata.append(
                {
                    "source": source,
                    "route": route,
                    "documents": len(rows),
                    "tokens": sum(r["content_length"] for r in rows),
                    "draws": len(sampled),
                }
            )
        cases, keys = [], []
        for report in reports:
            directory = run.folder(plan, report["slot"])
            decisions = {
                r["case_id"]: r for r in read_rows(directory / "decisions.parquet")
            }
            wanted = selected & decisions.keys()
            if not wanted:
                continue
            for row in read_rows(directory / "documents.parquet"):
                cid = old.case_id(row)
                if cid in wanted:
                    cases.append({"case_id": cid, **row})
                    keys.append(decisions[cid])
        names = (
            "review-packet.json",
            "draws.json",
            "audit-key.json",
            "part-audits.json",
        )
        _write_json(
            output / names[0],
            {
                "cases": cases,
                "scope": "Original texts; inspect sampled cl100k_base token locations in context. No model verdicts in this packet.",
            },
        )
        _write_json(
            output / names[1],
            {
                "strata": strata,
                "draws": selections,
                "scope": "Sampling with replacement, proportional to document token length within source/route. Retain repeated draws. Combine sources weighted by eligible token mass; do not average sources equally. Inference is limited to completed work, not the unsampled corpus. No independent ground-truth labels supplied.",
            },
        )
        _write_json(output / names[2], keys)
        _write_json(output / names[3], reports)
        by_source = {}
        for source in run.SOURCES:
            group = [r for r in reports if r["source"] == source]
            by_source[source] = {
                k: sum(r[k] for r in group)
                for k in (
                    "records",
                    "input_tokens",
                    "candidate_records",
                    "candidate_tokens",
                    "unresolved_documents",
                )
            }
            by_source[source]["parts"] = len(group)
            by_source[source]["generation_seconds_latest_invocations"] = sum(
                g["generation_seconds"]
                for r in group
                for g in r["generation_this_invocation"]
            )
            by_source[source]["worker_wall_seconds_latest_invocations"] = sum(
                r["wall_seconds_this_invocation"] for r in group
            )
        result = {
            "audit_complete": not errors,
            "plan_sha256": plan_sha,
            "all_planned_work_complete": not errors
            and not missing
            and not any(r["unresolved_documents"] for r in reports),
            "completed_parts": len(reports),
            "planned_parts": len(plan["parts"]),
            "unprocessed_slots": missing,
            "errors": errors,
            "by_source": by_source,
            "candidate_exact_unique_tokens_in_this_run": unique_tokens,
            "files": {n: _sha256(output / n) for n in names},
            "quality_review_complete": False,
            "release_ready": False,
            "scope": "Provisional candidates. Single-pass processing and sampled review do not establish accuracy or final retained additions. Runtime counters are latest invocations and may undercount resumed work.",
        }
        _write_json(output / "summary.json", result)
        temporary = output / "review-bundle.tar.gz.partial"
        with tarfile.open(temporary, "w:gz") as archive:
            for name in (*names, "summary.json"):
                archive.add(output / name, arcname=name)
            archive.add(path, arcname="plan.json")
        bundle = output / "review-bundle.tar.gz"
        temporary.replace(bundle)
        _write_json(
            output / "bundle-summary.json",
            {"bytes": bundle.stat().st_size, "sha256": _sha256(bundle)},
        )
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args(argv)
    report = finalize(args.plan)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "unprocessed_slots"}, indent=2
        )
    )
    return 0 if report["audit_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

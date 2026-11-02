"""Reparse production raw responses, prove export coverage and sample for review.

CPU-only. Integrity/completeness is distinct from independently established
classification quality. This command never publishes or certifies a release.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import tarfile

import pyarrow.parquet as pq

from scripts.youtube.download import _directory_lock, _write_json
from scripts.youtube.profile import _sha256
from scripts.youtube_filter.rubric import LABELS
from . import evaluate, production as prod, score, score_partitions


def read_rows(path):
    with pq.ParquetFile(path) as f:
        return f.read().to_pylist()


def audit_part(folder, plan, plan_sha, slot):
    part = plan["parts"][slot]
    packet = score.load_packet(folder / "work/packet.json")
    expected = score_partitions.part_packet(
        Path(plan["input_dir"]),
        {k: v for k, v in part.items() if k != "index"},
        plan["manifest_sha256"],
    )
    if packet != expected:
        raise ValueError("Export packet differs from prepared source")
    documents = read_rows(folder / "documents.parquet")
    if documents != prod.source_rows(packet):
        raise ValueError("Export changed or omitted source text/lineage")
    reports, spans = {}, []
    for stage in prod.STAGES:
        work = folder / "work" / stage
        config = json.loads((work / "run-config.json").read_text())
        expected_config = {
            **plan["settings"],
            "model": plan["model"]["model"],
            "model_revision": plan["model"]["revision"],
            "rubric_version": prod.STAGES[stage][0],
            "temperature": 0,
            "enable_thinking": False,
            "language_model_only": True,
            "enable_cuda_graphs": True,
        }
        if any(config.get(k) != v for k, v in expected_config.items()):
            raise ValueError("Scoring settings differ from frozen production plan")
        reports[stage] = evaluate.evaluate(folder / "work/packet.json", score_dir=work)
        spans.extend(prod.span_rows(packet, folder / "work", stage))
    decisions = prod.combine(packet, reports, plan_sha)
    if read_rows(folder / "decisions.parquet") != decisions:
        raise ValueError("Exported decisions disagree with reparsed raw responses")
    if read_rows(folder / "spans.parquet") != spans:
        raise ValueError("Exported span evidence disagrees with raw checkpoints")
    eligible = {r["case_id"] for r in decisions if r["provisional_route"] == "eligible"}
    candidates = [r for r in documents if prod.case_id(r) in eligible]
    if read_rows(folder / "candidates.parquet") != candidates:
        raise ValueError(
            "Candidate export differs from jointly eligible source records"
        )
    tokens = Counter()
    for row in decisions:
        tokens[row["provisional_route"]] += row["content_length"]
    generation = Counter()
    for span in spans:
        for attempt in json.loads(span["attempts_json"]):
            generation["attempts"] += 1
            generation["input_tokens"] += attempt["input_tokens"]
            generation["output_tokens"] += attempt["output_tokens"]
    return {
        "records": len(documents),
        "input_tokens": sum(r["content_length"] for r in documents),
        "routes": dict(Counter(r["provisional_route"] for r in decisions)),
        "token_routes": dict(tokens),
        "candidate_records": len(candidates),
        "candidate_tokens": sum(r["content_length"] for r in candidates),
        "unresolved_documents": sum(not r["scoring_complete"] for r in decisions),
        "spans": len(spans),
        "saved_generation": dict(generation),
    }


def finalize(plan_path):
    plan = prod.load_plan(plan_path)
    root = Path(plan["output_dir"])
    output = root / "review"
    output.mkdir(exist_ok=True)
    plan_sha = _sha256(plan_path)
    with _directory_lock(output):
        # An old successful report must not survive an interrupted new audit.
        _write_json(
            output / "summary.json",
            {"complete": False, "status": "auditing", "plan_sha256": plan_sha},
        )
        lineage = Path(plan["input_dir"]) / "lineage.parquet"
        if _sha256(lineage) != plan["lineage_sha256"]:
            raise ValueError("Prepared alias lineage changed")
        reports, errors, pool = [], [], defaultdict(list)
        frames = Counter()
        candidate_hashes = set()
        duplicate_candidate_copies = 0
        unique_candidate_tokens = 0
        for slot in range(len(plan["parts"])):
            try:
                summary = prod.checked_summary(plan_path, plan, slot)
                if not summary or not summary.get("workflow_finished"):
                    raise ValueError("Worker has not finished")
                folder = prod.part_dir(plan, slot)
                checked = audit_part(folder, plan, plan_sha, slot)
                if any(summary[k] != v for k, v in checked.items()):
                    raise ValueError("Worker summary does not match fresh audit")
                reports.append(
                    {
                        "slot": slot,
                        "source": plan["parts"][slot]["path"].split("/")[1],
                        "part_summary_sha256": _sha256(folder / "summary.json"),
                        **checked,
                    }
                )
                decisions = {
                    r["case_id"]: r for r in read_rows(folder / "decisions.parquet")
                }
                for row in read_rows(folder / "documents.parquet"):
                    decision = decisions[prod.case_id(row)]
                    route = decision["provisional_route"]
                    key = (row["source"], route)
                    frames[key] += 1
                    rank = prod.digest(
                        {"audit": "production-v1", "case_id": prod.case_id(row)}
                    )
                    pool[key].append((rank, row, decision))
                    pool[key].sort(key=lambda item: item[0])
                    del pool[key][plan["audit_per_stratum"] :]
                    if route == "eligible":
                        if row["content_hash"] in candidate_hashes:
                            duplicate_candidate_copies += 1
                        else:
                            candidate_hashes.add(row["content_hash"])
                            unique_candidate_tokens += row["content_length"]
            except (OSError, ValueError, KeyError) as error:
                errors.append({"slot": slot, "error": str(error)})
        cases, keys = [], []
        for group in sorted(pool):
            for _, row, decision in pool[group]:
                cases.append(
                    {
                        "case_id": prod.case_id(row),
                        "source": row["source"],
                        "text": row["text"],
                        "text_sha256": row["content_hash"],
                        "content_length": row["content_length"],
                        "unit": "complete_document",
                        "purpose": "candidate_scoring",
                        "lineage": {
                            k: row[k]
                            for k in (
                                "dataset_revision",
                                "source_shard",
                                "source_row",
                                "source_copies",
                            )
                        },
                    }
                )
                keys.append(decision)
        packet = {
            "version": 1,
            "purpose": "candidate_scoring",
            "labels": LABELS,
            "cases": cases,
        }
        packet["packet_sha256"] = prod.digest(packet)
        _write_json(output / "review-packet.json", packet)
        (output / "audit-key.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in keys)
        )
        _write_json(output / "part-audits.json", reports)
        complete = not errors and all(r["unresolved_documents"] == 0 for r in reports)
        by_source = {}
        for source in prod.SOURCES:
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
        result = {
            "complete": complete,
            "integrity_verified": not errors,
            "scoring_coverage_complete": complete,
            "plan_sha256": plan_sha,
            "completed_parts": len(reports),
            "expected_parts": len(plan["parts"]),
            "errors": errors,
            "by_source": by_source,
            "candidate_records_before_cross_source_dedup": sum(
                r["candidate_records"] for r in reports
            ),
            "candidate_tokens_before_cross_source_dedup": sum(
                r["candidate_tokens"] for r in reports
            ),
            "candidate_exact_unique_tokens_in_this_batch": unique_candidate_tokens,
            "duplicate_candidate_copies_in_this_batch": duplicate_candidate_copies,
            "review_sample_documents": len(cases),
            "review_strata": [
                {"source": s, "route": r, "population": n, "sampled": len(pool[(s, r)])}
                for (s, r), n in sorted(frames.items())
            ],
            "files": {
                n: _sha256(output / n)
                for n in ("review-packet.json", "audit-key.jsonl", "part-audits.json")
            },
            "quality_review_complete": False,
            "release_ready": False,
            "selection_decision": None,
            "scope": "Integrity and parsed coverage only. Review packet has no model verdicts; audit-key contains them separately. Complete source texts, not excerpts. Counts describe provisional candidates in completed parts, not final additions. No baseline or near deduplication, attribution or publication clearance.",
        }
        _write_json(output / "summary.json", result)
        temp = output / "review-bundle.tar.gz.partial"
        with tarfile.open(temp, "w:gz") as archive:
            for name in (
                "summary.json",
                "review-packet.json",
                "audit-key.jsonl",
                "part-audits.json",
            ):
                archive.add(output / name, arcname=name)
            archive.add(plan_path, arcname="plan.json")
        target = output / "review-bundle.tar.gz"
        temp.replace(target)
        _write_json(
            output / "bundle-summary.json",
            {
                "path": target.name,
                "bytes": target.stat().st_size,
                "sha256": _sha256(target),
            },
        )
        return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    a = p.parse_args(argv)
    result = finalize(a.plan)
    print(json.dumps(result, indent=2))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Export evidence from completed production work without new model inference.

Keep this outside scripts/curation: its Python files are bound by the completed
production plan. No thresholds, decisions, source text or old artifacts change.
"""

import argparse
from collections import Counter, defaultdict
import json
import logging
from pathlib import Path
import tarfile

from scripts.curation import production as prod, production_audit as audit, score
from scripts.youtube.download import _directory_lock, _write_json
from scripts.youtube.profile import _sha256

LOG = logging.getLogger(__name__)


def checked_review(plan_path, plan):
    root = Path(plan["output_dir"])
    review = root / "review"
    summary = json.loads((review / "summary.json").read_text())
    if (not summary.get("complete") or not summary.get("integrity_verified")
            or summary["plan_sha256"] != _sha256(plan_path)
            or summary["completed_parts"] != len(plan["parts"])
            or summary["expected_parts"] != len(plan["parts"])):
        raise ValueError("Completed, bound production audit is required")
    for name, checksum in summary["files"].items():
        if name not in {"audit-key.jsonl", "part-audits.json", "review-packet.json"}:
            raise ValueError("Unexpected review file")
        if _sha256(review / name) != checksum:
            raise ValueError("Review artifact checksum mismatch")
    if set(summary["files"]) != {"audit-key.jsonl", "part-audits.json", "review-packet.json"}:
        raise ValueError("Incomplete review manifest")
    packet = score.load_packet(review / "review-packet.json")
    keys = [json.loads(line) for line in (review / "audit-key.jsonl").read_text().splitlines()]
    if len(keys) != len(packet["cases"]) or len({k["case_id"] for k in keys}) != len(keys):
        raise ValueError("Review keys do not cover sample")
    return summary, packet, {k["case_id"]: k for k in keys}


def export(plan_path, output):
    plan_path, output = plan_path.resolve(), output.resolve()
    plan = prod.load_plan(plan_path)
    for protected in (Path(plan["output_dir"]), Path(plan["input_dir"])):
        protected = protected.resolve()
        if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("Diagnostics must be separate from production and source inputs")
    summary, packet, sample_keys = checked_review(plan_path, plan)
    root = Path(plan["output_dir"])
    old_parts = json.loads((root / "review/part-audits.json").read_text())
    if len(old_parts) != len(plan["parts"]) or {p["slot"] for p in old_parts} != set(range(len(plan["parts"]))):
        raise ValueError("Part audit coverage mismatch")
    old_parts = {p["slot"]: p for p in old_parts}
    binding = {
        "version": "production-evidence-v1",
        "plan_sha256": _sha256(plan_path),
        "review_summary_sha256": _sha256(root / "review/summary.json"),
        "exporter_sha256": _sha256(Path(__file__)),
        "production_code": plan["code"],
        "output_dir": str(output),
        "selection": "Existing source/route sample plus every jointly eligible YouTube document in this bounded batch",
    }
    output.mkdir(parents=True, exist_ok=True)
    with _directory_lock(output):
        config = output / "run-config.json"
        if config.exists() and json.loads(config.read_text()) != binding:
            raise ValueError("Diagnostics binding changed; use a new output directory")
        _write_json(config, binding)
        _write_json(output / "summary.json", {"complete": False, "status": "verifying"})
        if _sha256(Path(plan["input_dir"]) / "lineage.parquet") != plan["lineage_sha256"]:
            raise ValueError("Prepared lineage changed")
        sampled = {c["case_id"]: c for c in packet["cases"]}
        found = set()
        cases, evidence, runtimes = [], [], []
        pairs, pair_tokens, dimensions, concern_docs = Counter(), Counter(), Counter(), Counter()
        all_youtube_candidates = 0
        for slot in range(len(plan["parts"])):
            folder = prod.part_dir(plan, slot)
            saved = prod.checked_summary(plan_path, plan, slot)
            if not saved or not saved.get("complete"):
                raise ValueError(f"Slot {slot} is incomplete")
            checked = audit.audit_part(folder, plan, binding["plan_sha256"], slot)
            previous = old_parts[slot]
            if (_sha256(folder / "summary.json") != previous["part_summary_sha256"]
                    or any(saved[k] != v or previous[k] != v for k, v in checked.items())):
                raise ValueError("Fresh verification differs from completed audit")
            decisions = {r["case_id"]: r for r in audit.read_rows(folder / "decisions.parquet")}
            span_index = defaultdict(list)
            for span in audit.read_rows(folder / "spans.parquet"):
                span_index[span["content_hash"]].append(span)
            for row in audit.read_rows(folder / "documents.parquet"):
                cid = prod.case_id(row)
                decision = decisions[cid]
                source = row["source"]
                pair = (source, decision["screen_route"], decision["critic_route"], decision["provisional_route"])
                pairs[pair] += 1
                pair_tokens[pair] += row["content_length"]
                spans = span_index[row["content_hash"]]
                concerns = set()
                for span in spans:
                    for dimension in ("security_relevance", "technical_substance", "text_usability", "text_language", "mixed_content"):
                        dimensions[(source, span["stage"], dimension, span[dimension])] += 1
                    concerns.update((span["stage"], name) for name in span["quality_concerns"])
                for stage, name in concerns:
                    concern_docs[(source, stage, name)] += 1
                youtube_candidate = source == "youtube-commons" and decision["provisional_route"] == "eligible"
                all_youtube_candidates += youtube_candidate
                if cid not in sampled and not youtube_candidate:
                    continue
                if cid in sampled:
                    if row["text"] != sampled[cid]["text"] or decision != sample_keys[cid]:
                        raise ValueError("Sample differs from verified source or decisions")
                    found.add(cid)
                case = {
                    "case_id": cid, "source": source, "text": row["text"],
                    "text_sha256": row["content_hash"], "content_length": row["content_length"],
                    "unit": "complete_document", "purpose": "candidate_scoring",
                    "lineage": {k: row[k] for k in ("dataset_revision", "source_shard", "source_row", "source_copies")},
                }
                cases.append(case)
                evidence.append({
                    "case_id": cid, "slot": slot, "decision": decision,
                    "in_original_sample": cid in sampled,
                    "spans": [{**{k: v for k, v in s.items() if k != "attempts_json"},
                               "attempts": json.loads(s["attempts_json"])} for s in spans],
                })
            runtimes.append({
                "slot": slot, "source": previous["source"],
                "saved_generation": checked["saved_generation"],
                "generation_latest_call": saved["generation_latest_call"],
                "stage_wall_seconds_this_invocation": saved["stage_wall_seconds_this_invocation"],
                "caution": "Latest-invocation wall time can include checkpoint reuse; it is not total GPU time.",
            })
            LOG.info("Verified slot %s; %s documents", slot, checked["records"])
        if found != set(sampled):
            raise ValueError("Missing original review cases")
        if all_youtube_candidates != summary["by_source"].get("youtube-commons", {}).get("candidate_records", 0):
            raise ValueError("YouTube candidate coverage differs")
        cases.sort(key=lambda c: c["case_id"])
        if len({c["case_id"] for c in cases}) != len(cases):
            raise ValueError("Repeated evidence case")
        new_packet = {**{k: v for k, v in packet.items() if k not in {"cases", "packet_sha256"}}, "cases": cases}
        new_packet["packet_sha256"] = prod.digest(new_packet)
        _write_json(output / "review-packet.json", new_packet)
        with (output / "evidence.jsonl").open("w") as handle:
            for item in sorted(evidence, key=lambda r: r["case_id"]):
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        _write_json(output / "diagnostics.json", {
            "route_pairs": [{"source": k[0], "screen": k[1], "critic": k[2], "combined": k[3],
                             "documents": n, "tokens": pair_tokens[k]} for k, n in sorted(pairs.items())],
            "span_dimension_counts": [{"source": k[0], "stage": k[1], "dimension": k[2], "label": k[3],
                                       "spans": n} for k, n in sorted(dimensions.items())],
            "concern_document_counts": [{"source": k[0], "stage": k[1], "concern": k[2], "documents": n}
                                        for k, n in sorted(concern_docs.items())],
            "runtime": runtimes,
            "scope": "Concern counts overlap. Dimensions count spans, route pairs count documents. No causal attribution or classifier accuracy estimate.",
        })
        names = ("run-config.json", "review-packet.json", "evidence.jsonl", "diagnostics.json")
        result = {
            "complete": True, "inference_performed": False, "new_model_tokens": 0,
            "plan_sha256": binding["plan_sha256"], "original_sample_documents": len(sampled),
            "all_youtube_candidate_documents": all_youtube_candidates,
            "exported_documents": len(cases), "exported_source_tokens": sum(c["content_length"] for c in cases),
            "production_candidate_tokens": summary["candidate_tokens_before_cross_source_dedup"],
            "files": {n: _sha256(output / n) for n in names},
            "quality_review_complete": False, "release_ready": False, "selection_decision": None,
        }
        _write_json(output / "summary.json", result)
        temporary = output / "review-bundle.tar.gz.partial"
        with tarfile.open(temporary, "w:gz") as archive:
            for name in (*names, "summary.json"):
                archive.add(output / name, arcname=name)
        bundle = output / "review-bundle.tar.gz"
        temporary.replace(bundle)
        _write_json(output / "bundle-summary.json", {"path": bundle.name, "bytes": bundle.stat().st_size,
                                                      "sha256": _sha256(bundle)})
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(export(args.plan, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

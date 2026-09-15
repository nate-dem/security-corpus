"""Bounded source-only recheck of the successful 27B comparison.

Review every provisionally eligible/review item, including unparsed first-pass
items, plus a deterministic diagnostic sample of rejections and all controls.
No source is deleted, no final selection is made, and no full-corpus job starts.
"""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path

from scripts.youtube.download import _directory_lock, _write_json
from scripts.youtube.profile import _sha256
from . import critic_v2, cuda_preflight, evaluate, first_batch, policy, runtime, score

HERE = Path(__file__).resolve().parent
PACKETS = (
    "development-v2",
    "accepted-audit-v1",
    "part-001785",
    "part-002686",
    "part-009211",
)


def subset_packet(packet, comparisons, excluded_per_source):
    """Allocate a bounded diagnostic workload; never treat unsampled rows as drops."""
    if excluded_per_source < 1:
        raise ValueError("The diagnostic rejection sample must be positive")
    by_id = {r["case_id"]: r for r in comparisons}
    if len(by_id) != len(comparisons) or set(by_id) != {
        c["case_id"] for c in packet["cases"]
    }:
        raise ValueError("Comparison and packet coverage differ")
    if any(
        r["model_route"] not in {"eligible", "review_required", "exclude"}
        for r in comparisons
    ):
        raise ValueError("Unknown first-pass route")
    selected = {k for k, r in by_id.items() if r["model_route"] != "exclude"}
    for source in sorted({c["source"] for c in packet["cases"]}):
        rejected = [
            c
            for c in packet["cases"]
            if c["source"] == source and by_id[c["case_id"]]["model_route"] == "exclude"
        ]
        rejected.sort(
            key=lambda c: hashlib.sha256(
                f"quality-pass-v1-rejections:{c['case_id']}".encode()
            ).hexdigest()
        )
        selected.update(c["case_id"] for c in rejected[:excluded_per_source])
    result = {
        "version": 1,
        "purpose": "candidate_scoring",
        "labels": packet["labels"],
        "cases": [c for c in packet["cases"] if c["case_id"] in selected],
        "prior_packet_sha256": packet["packet_sha256"],
        "selection_decision": None,
    }
    result["packet_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return result


def load_plan(prior, excluded_per_source=12):
    model = json.loads((HERE / "next_models.json").read_text())["models"][1]
    summary = json.loads((prior / "summary.json").read_text())
    config = json.loads((prior / "run-config.json").read_text())
    if (
        summary.get("complete") is not True
        or summary.get("selection_decision") is not None
        or summary.get("model") != model
        or config.get("model") != model
        or summary["run_config_sha256"] != _sha256(prior / "run-config.json")
        or [r["name"] for r in summary["runs"]] != list(PACKETS)
    ):
        raise ValueError("Expected the completed, bound 27B first-batch-v4 comparison")
    jobs, bindings = [], {}
    for name in PACKETS:
        packet_path = prior / "packets" / f"{name}.json"
        reference = prior / "packets" / f"{name}-annotations.json"
        ref = reference if reference.is_file() else None
        if (name.startswith("part-") and ref) or (
            not name.startswith("part-") and not ref
        ):
            raise ValueError("Control/reference packet mismatch")
        first = evaluate.evaluate(packet_path, ref, prior / "runs" / name)
        recorded = next(r for r in summary["runs"] if r["name"] == name)
        if (
            first["model"] != model["model"]
            or first["model_revision"] != model["revision"]
            or first["run_config_sha256"] != recorded["run_config_sha256"]
        ):
            raise ValueError(
                "First-pass model/configuration differs from the bound summary"
            )
        # Reparse raw outputs and verify input hashes; cached route summaries are
        # not authoritative. Historical evaluator hashes may differ from today.
        packet = score.load_packet(packet_path)
        selected = (
            packet
            if ref
            else subset_packet(packet, first["comparisons"], excluded_per_source)
        )
        by_id = {r["case_id"]: r for r in first["comparisons"]}
        bindings[name] = {
            "parent_packet_sha256": packet["packet_sha256"],
            "selected_packet_sha256": selected["packet_sha256"],
            "first_pass_config_sha256": first["run_config_sha256"],
            "first_pass_summary_sha256": first["score_summary_sha256"],
            "reference_sha256": first["reference_sha256"],
            "cases": len(selected["cases"]),
            "tokens": sum(c["content_length"] for c in selected["cases"]),
            "first_routes": dict(
                Counter(by_id[c["case_id"]]["model_route"] for c in selected["cases"])
            ),
        }
        jobs.append((name, selected, ref, by_id))
    binding = {
        "version": "quality-pass-v1",
        "model": model,
        "prior_summary_sha256": _sha256(prior / "summary.json"),
        "runner_sha256": _sha256(Path(__file__)),
        "score_sha256": _sha256(Path(score.__file__)),
        "evaluator_sha256": _sha256(Path(evaluate.__file__)),
        "policy_sha256": _sha256(Path(policy.__file__)),
        "critic_sha256": _sha256(Path(critic_v2.__file__)),
        "excluded_per_source": excluded_per_source,
        "packets": bindings,
        "scope": "Diagnostic recheck. Rejection sampling is not a scope filter; unsampled texts remain unchanged. References are assistant development labels, not independent ground truth.",
    }
    return binding, jobs


def compare(first_by_id, second):
    rows = []
    for row in second["comparisons"]:
        first = first_by_id[row["case_id"]]
        a, b = first["model_route"], row["model_route"]
        rows.append(
            {
                "case_id": row["case_id"],
                "source": row["source"],
                "content_length": row["content_length"],
                "first_route": a,
                "second_route": b,
                "diagnostic_route": a if a == b else "review_required",
                "assistant_route": row["assistant_route"],
                "selection_decision": None,
            }
        )
    return {
        "cases": len(rows),
        "comparisons": rows,
        "first_vs_second": dict(
            Counter(f"{r['first_route']} -> {r['second_route']}" for r in rows)
        ),
        "diagnostic_routes": dict(Counter(r["diagnostic_route"] for r in rows)),
        "scope": "Agreement is diagnostic. Same-model calls have correlated errors; no selection or accuracy certificate.",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--excluded-per-source",
        type=int,
        default=12,
        help="Diagnostic workload size, not a filtering threshold",
    )
    p.add_argument("--check-env", action="store_true")
    a = p.parse_args(argv)
    prior, output = a.prior_dir.resolve(), a.output_dir.resolve()
    if prior == output or prior.is_relative_to(output) or output.is_relative_to(prior):
        p.error("Keep outputs separate from the first-pass inputs")
    binding, jobs = load_plan(prior, a.excluded_per_source)
    if a.check_env:
        print(json.dumps(binding, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    with _directory_lock(output):
        config = {
            **binding,
            "cuda_compiler": cuda_preflight.configure_and_check(),
            "sampler_preflight": cuda_preflight.check_sampler(),
            "installed_packages": {
                d.metadata["Name"]: d.version
                for d in importlib.metadata.distributions()
                if d.metadata["Name"]
            },
        }
        config_path = output / "run-config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError("Run binding changed; use a new output directory")
        _write_json(config_path, config)
        _write_json(output / "summary.json", {"complete": False, "status": "running"})
        for name in ("packets", "runs", "evaluations"):
            (output / name).mkdir(exist_ok=True)
        factory = runtime.ResidentModel()
        reports = []
        for name, packet, ref, first_by_id in jobs:
            path = output / "packets" / f"{name}.json"
            _write_json(path, packet)
            reference = None
            if ref:
                reference = output / "packets" / f"{name}-annotations.json"
                reference.write_bytes(ref.read_bytes())
            model = binding["model"]
            status = score.main(
                [
                    "--packet",
                    str(path),
                    "--output-dir",
                    str(output / "runs" / name),
                    "--rubric-version",
                    "critic-v2",
                    "--model",
                    model["model"],
                    "--model-revision",
                    model["revision"],
                    "--tensor-parallel-size",
                    "1",
                    "--batch-size",
                    "32",
                    "--language-model-only",
                    "--local-files-only",
                    "--enable-cuda-graphs",
                    "--max-output-tokens",
                    "1536",
                ],
                model_factory=factory,
            )
            if status not in (0, 2):
                raise RuntimeError(f"Second-pass scoring failed: {status}")
            report = evaluate.evaluate(path, reference, output / "runs" / name)
            _write_json(output / "evaluations" / f"{name}.json", report)
            joined = compare(first_by_id, report)
            _write_json(output / "evaluations" / f"{name}-comparison.json", joined)
            reports.append(
                {
                    "name": name,
                    **{
                        k: report[k]
                        for k in (
                            "cases",
                            "model_routes",
                            "scoring_coverage_complete",
                            "action_agreements",
                            "model_eligible_reference_not_eligible",
                            "reference_eligible_model_not_eligible",
                            "generation",
                        )
                    },
                    "first_vs_second": joined["first_vs_second"],
                    "diagnostic_routes": joined["diagnostic_routes"],
                }
            )
        complete = all(r["scoring_coverage_complete"] for r in reports)
        _write_json(
            output / "summary.json",
            {
                "complete": complete,
                "workflow_finished": True,
                "scoring_coverage_complete": complete,
                "runs": reports,
                "model": binding["model"],
                "run_config_sha256": _sha256(config_path),
                "selection_decision": None,
                "production_accuracy_established": False,
                "scope": binding["scope"],
            },
        )
        first_batch.bundle(output)
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())

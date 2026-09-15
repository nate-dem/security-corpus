"""Reassess only documents with unresolved quality-pass responses.

Keep original outputs immutable. Compact JSON removes the observed unbounded
inter-field whitespace path; it does not change the source or quality rubric.
An explicit manifest links each retry to the original packet and raw decisions.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from scripts.youtube.download import _directory_lock, _write_json
from scripts.youtube.profile import _sha256
from . import (
    critic_v2,
    cuda_preflight,
    evaluate,
    first_batch,
    quality_pass,
    runtime,
    score,
)


def load_plan(prior):
    summary = json.loads((prior / "summary.json").read_text())
    model = json.loads((quality_pass.HERE / "next_models.json").read_text())["models"][
        1
    ]
    if (
        summary.get("workflow_finished") is not True
        or summary.get("model") != model
        or summary["run_config_sha256"] != _sha256(prior / "run-config.json")
        or [r["name"] for r in summary["runs"]] != list(quality_pass.PACKETS)
    ):
        raise ValueError("Expected a finished, bound 27B quality-pass-v1 workflow")
    jobs, binding = [], {}
    total_spans = 0
    for name in quality_pass.PACKETS:
        source = prior / "runs" / name
        packet_path = prior / "packets" / f"{name}.json"
        report = evaluate.evaluate(packet_path, score_dir=source)
        old_config = json.loads((source / "run-config.json").read_text())
        if (
            report["model"] != model["model"]
            or report["model_revision"] != model["revision"]
            or old_config["prompt_version"] != critic_v2.VERSION
            or old_config["max_output_tokens"] != 1536
            or old_config["max_model_len"] != 8192
            or old_config["focal_tokens"] != 4096
        ):
            raise ValueError(
                "Retry must match the original model, rubric and span budgets"
            )
        unresolved = {
            r["case_id"]
            for r in report["comparisons"]
            if any(s != "ok" for s in r["parse_statuses"])
        }
        requests = [
            json.loads(line)
            for line in (source / "requests.jsonl").read_text().splitlines()
        ]
        total_spans += len(requests)
        if not unresolved:
            continue
        original = score.load_packet(packet_path)
        cases = [c for c in original["cases"] if c["case_id"] in unresolved]
        packet = {
            "version": 1,
            "purpose": "candidate_scoring",
            "labels": original["labels"],
            "cases": cases,
            "prior_packet_sha256": original["packet_sha256"],
            "selection_decision": None,
        }
        packet["packet_sha256"] = hashlib.sha256(
            json.dumps(packet, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        hashes = {c["text_sha256"] for c in cases}
        selected_requests = [t for t in requests if t["content_hash"] in hashes]
        binding[name] = {
            "packet_sha256": packet["packet_sha256"],
            "prior_packet_sha256": original["packet_sha256"],
            "prior_config_sha256": report["run_config_sha256"],
            "prior_summary_sha256": _sha256(source / "summary.json"),
            "prior_requests_sha256": _sha256(source / "requests.jsonl"),
            "prior_decision_sha256": {
                t["key"]: _sha256(source / "decisions" / f"{t['key']}.json")
                for t in selected_requests
            },
            "expected_span_keys": [t["key"] for t in selected_requests],
            "expected_task_sha256": {
                t["key"]: t["task_sha256"] for t in selected_requests
            },
            "cases": len(cases),
            "tokens": sum(c["content_length"] for c in cases),
            "old_unresolved_spans": sum(
                s != "ok" for r in report["comparisons"] for s in r["parse_statuses"]
            ),
        }
        jobs.append((name, packet))
    return {
        "version": "quality-retry-v1",
        "model": model,
        "packets": binding,
        "prior_summary_sha256": _sha256(prior / "summary.json"),
        "prior_total_spans": total_spans,
        "runner_sha256": _sha256(Path(__file__)),
        "score_sha256": _sha256(Path(score.__file__)),
        "evaluator_sha256": _sha256(Path(evaluate.__file__)),
        "scope": "Separate retry assessments; original outputs remain intact. No final corpus selection. Whole unresolved documents are rechecked without shortening their text.",
    }, jobs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--check-env", action="store_true")
    a = p.parse_args(argv)
    prior, output = a.prior_dir.resolve(), a.output_dir.resolve()
    if prior == output or prior.is_relative_to(output) or output.is_relative_to(prior):
        p.error("Keep retry output separate from prior artifacts")
    binding, jobs = load_plan(prior)
    if a.check_env:
        print(json.dumps(binding, indent=2))
        return 0
    if not jobs:
        print("No unresolved documents; no model loaded.")
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
            raise ValueError("Retry binding changed; choose a new output directory")
        _write_json(config_path, config)
        for name in ("packets", "runs", "evaluations"):
            (output / name).mkdir(exist_ok=True)
        _write_json(output / "summary.json", {"complete": False, "status": "running"})
        factory = runtime.ResidentModel()
        reports = []
        for name, packet in jobs:
            path = output / "packets" / f"{name}.json"
            _write_json(path, packet)
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
                    "--compact-json",
                ],
                model_factory=factory,
            )
            if status not in (0, 2):
                raise RuntimeError(f"Retry scoring failed: {status}")
            requests = [
                json.loads(line)
                for line in (output / "runs" / name / "requests.jsonl")
                .read_text()
                .splitlines()
            ]
            if [t["key"] for t in requests] != binding["packets"][name][
                "expected_span_keys"
            ] or {t["key"]: t["task_sha256"] for t in requests} != binding["packets"][
                name
            ]["expected_task_sha256"]:
                raise ValueError(
                    "Retry source spans or prompts changed; preserve results for inspection"
                )
            report = evaluate.evaluate(path, score_dir=output / "runs" / name)
            _write_json(output / "evaluations" / f"{name}.json", report)
            reports.append(
                {
                    "name": name,
                    **{
                        k: report[k]
                        for k in (
                            "cases",
                            "scoring_coverage_complete",
                            "model_routes",
                            "generation",
                        )
                    },
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
                "prior_outputs_modified": False,
                "scope": binding["scope"],
            },
        )
        first_batch.bundle(output)
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())

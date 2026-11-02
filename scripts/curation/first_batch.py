"""Compare pinned models on controls and three complete prepared work units.

One randomly ordered cluster (partition) per source measures operational behavior,
not representative corpus yield. Every selected document is scored in full. Raw
responses and provenance remain available; all selection decisions stay null.
"""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tarfile

from scripts.youtube.download import _directory_lock, _write_json
from scripts.youtube.profile import _sha256
from . import cuda_preflight, evaluate, runtime, score, score_partitions

HERE = Path(__file__).resolve().parent


def select_parts(summary):
    """Workload allocation, not a content/scope filter or yield sample."""
    selected = []
    for source in ("primus-fineweb", "redsage-cfw", "youtube-commons"):
        choices = [
            (i, p)
            for i, p in enumerate(summary["parts"])
            if p["path"].split("/")[1] == source
        ]
        if not choices:
            raise ValueError(f"No prepared partitions for {source}")
        i, part = min(
            choices,
            key=lambda item: hashlib.sha256(
                f"curation-first-batch-v1:{item[1]['path']}".encode()
            ).hexdigest(),
        )
        selected.append({"index": i, **part})
    return selected


def load_inputs(project, input_dir):
    prepared = score_partitions.load_prepared(input_dir)
    controls = [
        project / "reports/curation" / name
        for name in ("development-v2", "accepted-audit-v1")
    ]
    for path in controls:
        evaluate.evaluate(path / "packet.json", path / "assistant_annotations.json")
    models = json.loads((HERE / "next_models.json").read_text())["models"]
    binding = {
        "version": "curation-first-batch-v1",
        "manifest_sha256": _sha256(input_dir / "summary.json"),
        "parts": select_parts(prepared),
        "controls": {
            p.name: {
                n: _sha256(p / n) for n in ("packet.json", "assistant_annotations.json")
            }
            for p in controls
        },
        "models_sha256": _sha256(HERE / "next_models.json"),
        "runner_sha256": _sha256(Path(__file__)),
        "partition_reader_sha256": _sha256(Path(score_partitions.__file__)),
        "evaluator_sha256": _sha256(Path(evaluate.__file__)),
    }
    return binding, controls, models


def score_one(packet, reference, destination, model, factory):
    args = [
        "--packet",
        str(packet),
        "--output-dir",
        str(destination),
        "--rubric-version",
        "v4",
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
    ]
    status = score.main(args, model_factory=factory)
    if status not in (0, 2):
        raise RuntimeError(f"Scoring failed with status {status}")
    return evaluate.evaluate(packet, reference, destination)


def bundle(output):
    # This contains only the bounded experiment; never the input tree/model cache.
    target = output / "review-bundle.tar.gz"
    temporary = output / "review-bundle.tar.gz.partial"
    with tarfile.open(temporary, "w:gz") as archive:
        for name in (
            "run-config.json",
            "summary.json",
            "packets",
            "evaluations",
            "runs",
        ):
            archive.add(output / name, arcname=f"{output.name}/{name}")
    temporary.replace(target)
    _write_json(
        output / "bundle-summary.json",
        {
            "path": target.name,
            "bytes": target.stat().st_size,
            "sha256": _sha256(target),
        },
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-index", type=int, choices=(0, 1))
    p.add_argument("--check-env", action="store_true")
    a = p.parse_args(argv)
    project, root, output = (
        a.project.resolve(),
        a.input_dir.resolve(),
        a.output_dir.resolve(),
    )
    if output == root or output.is_relative_to(root) or root.is_relative_to(output):
        p.error("Input and output must be separate")
    binding, controls, models = load_inputs(project, root)
    if a.check_env:
        print(
            json.dumps(
                {
                    "input_manifest_verified": True,
                    "parts": binding["parts"],
                    "candidate_records_per_model": sum(
                        x["records"] for x in binding["parts"]
                    ),
                    "candidate_tokens_per_model": sum(
                        x["tokens"] for x in binding["parts"]
                    ),
                    "models": models,
                },
                indent=2,
            )
        )
        return 0
    if a.model_index is None:
        p.error("--model-index is required for inference")
    model = models[a.model_index]
    output = output / model["name"]
    for control in controls:
        if (
            output == control
            or output.is_relative_to(control)
            or control.is_relative_to(output)
        ):
            p.error("Output must be separate from control inputs")
    output.mkdir(parents=True, exist_ok=True)
    with _directory_lock(output):
        compiler = cuda_preflight.configure_and_check()
        sampler = cuda_preflight.check_sampler()
        environment = {
            d.metadata["Name"]: d.version
            for d in importlib.metadata.distributions()
            if d.metadata["Name"]
        }
        config = {
            **binding,
            "model": model,
            "installed_packages": environment,
            "cuda_compiler": compiler,
            "sampler_preflight": sampler,
        }
        config_path = output / "run-config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError("Run binding changed; choose a new output directory")
        _write_json(config_path, config)
        _write_json(output / "summary.json", {"complete": False, "status": "running"})
        for name in ("packets", "runs", "evaluations"):
            (output / name).mkdir(exist_ok=True)
        factory = runtime.ResidentModel()
        reports = []
        for control in controls:
            packet = output / "packets" / (control.name + ".json")
            reference = output / "packets" / (control.name + "-annotations.json")
            packet.write_bytes((control / "packet.json").read_bytes())
            reference.write_bytes((control / "assistant_annotations.json").read_bytes())
            report = score_one(
                packet, reference, output / "runs" / control.name, model, factory
            )
            _write_json(output / "evaluations" / (control.name + ".json"), report)
            reports.append((control.name, report))
        for part in binding["parts"]:
            index = part["index"]
            prepared_part = {k: v for k, v in part.items() if k != "index"}
            packet = score_partitions.part_packet(
                root, prepared_part, binding["manifest_sha256"]
            )
            path = output / "packets" / f"part-{index:06d}.json"
            _write_json(path, packet)
            name = path.stem
            report = score_one(path, None, output / "runs" / name, model, factory)
            _write_json(output / "evaluations" / (name + ".json"), report)
            reports.append((name, report))
        compact = []
        for name, report in reports:
            rows = report["comparisons"]
            token_routes = Counter()
            for row in rows:
                token_routes[row["model_route"]] += row["content_length"]
            compact.append(
                {
                    "name": name,
                    **{
                        k: report[k]
                        for k in (
                            "cases",
                            "model_routes",
                            "scoring_coverage_complete",
                            "generation",
                            "action_agreements",
                            "model_eligible_reference_not_eligible",
                            "reference_eligible_model_not_eligible",
                            "run_config_sha256",
                        )
                    },
                    "token_routes": dict(token_routes),
                }
            )
        _write_json(
            output / "summary.json",
            {
                "complete": True,
                "scoring_coverage_complete": all(
                    r["scoring_coverage_complete"] for _, r in reports
                ),
                "model": model,
                "run_config_sha256": _sha256(config_path),
                "runs": compact,
                "selection_decision": None,
                "production_accuracy_established": False,
                "scope": "Bounded model/workload comparison. Controls are assistant-reviewed development examples; one partition per source is a cluster sample, not a yield estimate. Token routes are candidates, not additions. No automatic winner or selection.",
            },
        )
        bundle(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

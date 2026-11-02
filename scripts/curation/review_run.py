"""GPU-only follow-up using transferred diagnostic packets; no bulk-prep dependency."""

import argparse
from collections import Counter
from pathlib import Path

from scripts.youtube.download import _write_json
from scripts.youtube.profile import _sha256
from . import evaluate, score
from .runtime import ResidentModel


def cascade(packet, screen, critic, output, reference=None):
    first = evaluate.evaluate(packet, reference, screen)
    second = evaluate.evaluate(packet, reference, critic)
    by_id = {r["case_id"]: r for r in second["comparisons"]}
    rows = []
    for row in first["comparisons"]:
        other = by_id[row["case_id"]]
        a, b = row["model_route"], other["model_route"]
        route = a if a == b else "review_required"
        rows.append(
            {
                "case_id": row["case_id"],
                "source": row["source"],
                "screen_route": a,
                "critic_route": b,
                "provisional_route": route,
                "assistant_route": row["assistant_route"],
                "selection_decision": None,
            }
        )
    result = {
        "complete": True,
        "packet_sha256": first["packet_sha256"],
        "screen_config_sha256": first["run_config_sha256"],
        "critic_config_sha256": second["run_config_sha256"],
        "evaluator_sha256": _sha256(Path(evaluate.__file__)),
        "runner_sha256": _sha256(Path(__file__)),
        "scoring_coverage_complete": first["scoring_coverage_complete"]
        and second["scoring_coverage_complete"],
        "reference_kind": first["reference_kind"],
        "independent_ground_truth": False,
        "production_accuracy_established": False,
        "cases": len(rows),
        "routes": dict(Counter(r["provisional_route"] for r in rows)),
        "by_source": {
            s: dict(Counter(r["provisional_route"] for r in rows if r["source"] == s))
            for s in sorted({r["source"] for r in rows})
        },
        "assistant_eligible_recovered": sum(
            r["assistant_route"] == "eligible" and r["provisional_route"] == "eligible"
            for r in rows
        )
        if reference
        else None,
        "provisional_eligible_reference_not_eligible": [
            r["case_id"]
            for r in rows
            if r["provisional_route"] == "eligible"
            and r["assistant_route"] not in (None, "eligible")
        ],
        "scope": "Source-only screening and reviewer agreement is diagnostic, not independent accuracy or final selection. Disagreements and parse failures remain review items.",
        "comparisons": rows,
    }
    _write_json(output, result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("screen", "review"), required=True)
    p.add_argument("--reviewed-input", type=Path, required=True)
    p.add_argument("--additional-input", type=Path, required=True)
    p.add_argument("--prior-screen", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args(argv)
    packet = a.reviewed_input / "packet.json"
    reference = a.reviewed_input / "assistant_annotations.json"
    additional = a.additional_input / "packet.json"
    for f in (packet, additional):
        score.load_packet(f)
    # Validate/reparse the successful prior 8B run before using its judgments.
    evaluate.evaluate(packet, reference, a.prior_screen)
    root = a.output_dir.resolve()
    if any(
        root == d.resolve()
        or root.is_relative_to(d.resolve())
        or d.resolve().is_relative_to(root)
        for d in (a.reviewed_input, a.additional_input, a.prior_screen)
    ):
        p.error("Output must be separate from all inputs")
    root.mkdir(parents=True, exist_ok=True)
    factory = ResidentModel()
    size = 8 if a.phase == "screen" else 32
    revision = (
        "b968826d9c46dd6066d109eabc6255188de91218"
        if size == 8
        else "9216db5781bf21249d130ec9da846c4624c16137"
    )
    runs = (
        [(additional, "additional-screen", None)]
        if size == 8
        else [
            (packet, "development-critic", reference),
            (additional, "additional-critic", None),
        ]
    )
    for source, name, ref in runs:
        status = score.main(
            [
                "--packet",
                str(source),
                "--output-dir",
                str(root / name),
                "--rubric-version",
                "v3" if size == 8 else "critic-v1",
                "--model",
                f"Qwen/Qwen3-{size}B",
                "--model-revision",
                revision,
                "--tensor-parallel-size",
                "1" if size == 8 else "2",
            ],
            model_factory=factory,
        )
        if status not in (0, 2):
            return status
        report = evaluate.evaluate(source, ref, root / name)
        _write_json(root / (name + "-evaluation.json"), report)
    if size == 32:
        cascade(
            packet,
            a.prior_screen,
            root / "development-critic",
            root / "development-cascade.json",
            reference,
        )
        cascade(
            additional,
            root / "additional-screen",
            root / "additional-critic",
            root / "additional-cascade.json",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

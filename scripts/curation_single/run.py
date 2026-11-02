"""Resumable single-pass work, preserved raw responses and source-bound exports."""

import argparse
from collections import Counter, defaultdict
import json
import logging
from pathlib import Path
import time

import pyarrow as pa

from scripts.curation import (
    production as old,
    score_partitions,
    runtime,
    cuda_preflight,
)
from scripts.curation.production_audit import read_rows
from scripts.youtube.download import _directory_lock, _write_json
from scripts.youtube.profile import _sha256
from scripts.youtube_filter import score as engine
from . import rubric, score

HERE = Path(__file__).resolve().parent
VERSION = "curation-single-pass-v1"
SOURCES = ("youtube-commons", "primus-fineweb", "redsage-cfw")
FILES = (
    "documents.parquet",
    "candidates.parquet",
    "decisions.parquet",
    "spans.parquet",
)
DECISIONS = pa.schema(
    [
        ("case_id", pa.string()),
        ("source", pa.string()),
        ("content_hash", pa.string()),
        ("content_length", pa.int64()),
        ("route", pa.string()),
        ("scoring_complete", pa.bool_()),
        ("run_config_sha256", pa.string()),
        ("plan_sha256", pa.string()),
        ("selection_decision", pa.bool_()),
    ]
)
SPANS = pa.schema(
    [
        ("key", pa.string()),
        ("content_hash", pa.string()),
        ("start", pa.int64()),
        ("end", pa.int64()),
        ("task_sha256", pa.string()),
        ("model", pa.string()),
        ("model_revision", pa.string()),
        ("prompt_version", pa.string()),
        ("run_config_sha256", pa.string()),
        ("parse_status", pa.string()),
        ("domain", pa.string()),
        ("quality", pa.string()),
        ("language", pa.string()),
        ("reason", pa.string()),
        ("attempts_json", pa.large_string()),
    ]
)


def code_hashes():
    return {
        **old.code_hashes(),
        **{
            str(p.relative_to(HERE.parent.parent)): _sha256(p)
            for p in sorted(HERE.iterdir())
            if p.suffix in {".py", ".sh", ".sbatch"}
        },
    }


def make_plan(root, output):
    root, output = root.resolve(), output.resolve()
    if root == output or root.is_relative_to(output) or output.is_relative_to(root):
        raise ValueError("Keep output separate from prepared inputs")
    manifest = score_partitions.load_prepared(root)
    groups = {}
    for source in SOURCES:
        group = [
            {"index": i, **p}
            for i, p in enumerate(manifest["parts"])
            if p["path"].split("/")[1] == source
        ]
        group.sort(key=lambda p: old.digest({"single-v1": p["path"]}))
        # Spread the first six transcript parts over different raw shards.
        ordered, rest, seen = [], [], set()
        for part in group:
            shard = str(Path(part["path"]).parent)
            if shard in seen:
                rest.append(part)
            else:
                ordered.append(part)
                seen.add(shard)
        groups[source] = ordered + rest
    if len(groups[SOURCES[0]]) < 6 or any(not groups[s] for s in SOURCES[1:]):
        raise ValueError(
            "Need six transcript shards and both web sources for first allocation"
        )
    parts = groups[SOURCES[0]][:6] + [groups[s][0] for s in SOURCES[1:]]
    parts += groups[SOURCES[0]][6:] + groups[SOURCES[1]][1:] + groups[SOURCES[2]][1:]
    if len(parts) != len(manifest["parts"]):
        raise ValueError("Unrecognized prepared source")
    plan = {
        "version": VERSION,
        "input_dir": str(root),
        "output_dir": str(output),
        "manifest_sha256": _sha256(root / "summary.json"),
        "lineage_sha256": manifest["index"]["lineage_sha256"],
        "parts": parts,
        "model": json.loads((old.first_batch.HERE / "next_models.json").read_text())[
            "models"
        ][1],
        "code": code_hashes(),
        "settings": {
            "tensor_parallel_size": 1,
            "batch_size": 32,
            "focal_tokens": 4096,
            "max_model_len": 8192,
            "max_output_tokens": 256,
        },
        "scope": "One broad utility/relevance pass; no second critic. This manifest inventories all sources; the submitted work list restricts inference to the explicitly requested source (YouTube by default). Whole-document candidates precede deduplication and release checks; unresolved or mixed span routes stay reviewable.",
    }
    output.mkdir(parents=True, exist_ok=True)
    with _directory_lock(output):
        path = output / "plan.json"
        if path.exists() and json.loads(path.read_text()) != plan:
            raise ValueError(
                "Plan changed; preserve checkpoints and use another output"
            )
        _write_json(path, plan)
    return plan


def load_plan(path):
    plan = json.loads(path.read_text())
    if (
        plan["version"] != VERSION
        or plan["code"] != code_hashes()
        or path.resolve() != Path(plan["output_dir"]) / "plan.json"
    ):
        raise ValueError("Plan code or output binding changed")
    root = Path(plan["input_dir"])
    if _sha256(root / "summary.json") != plan["manifest_sha256"]:
        raise ValueError("Prepared manifest changed")
    manifest = score_partitions.load_prepared(root)
    if {p["index"] for p in plan["parts"]} != set(range(len(manifest["parts"]))) or len(
        plan["parts"]
    ) != len(manifest["parts"]):
        raise ValueError("Plan coverage differs from input")
    for part in plan["parts"]:
        if {k: v for k, v in part.items() if k != "index"} != manifest["parts"][
            part["index"]
        ]:
            raise ValueError("Plan part changed")
    return plan


def folder(plan, slot):
    if not 0 <= slot < len(plan["parts"]):
        raise ValueError("Slot outside plan")
    return Path(plan["output_dir"]) / "parts" / f"{slot:05d}"


def checked_summary(path, plan, slot):
    directory = folder(plan, slot)
    report = directory / "summary.json"
    if not report.exists():
        return None
    value = json.loads(report.read_text())
    if value["plan_sha256"] != _sha256(path):
        raise ValueError("Worker plan changed")
    if value.get("workflow_finished"):
        if set(value["files"]) != set(FILES):
            raise ValueError("Incomplete worker manifest")
        for name, digest in value["files"].items():
            if _sha256(directory / name) != digest:
                raise ValueError("Worker output checksum mismatch")
    return value


def audit_part(path, plan, slot, verify_export=True):
    directory = folder(plan, slot)
    work = directory / "work"
    packet = score.load_packet(work / "packet.json")
    expected = score_partitions.part_packet(
        Path(plan["input_dir"]),
        {k: v for k, v in plan["parts"][slot].items() if k != "index"},
        plan["manifest_sha256"],
    )
    if packet != expected:
        raise ValueError("Packet differs from prepared source")
    scored = work / "score"
    config = json.loads((scored / "run-config.json").read_text())
    summary = json.loads((scored / "summary.json").read_text())
    config_sha = _sha256(scored / "run-config.json")
    settings = {
        **plan["settings"],
        "model": plan["model"]["model"],
        "model_revision": plan["model"]["revision"],
        "prompt_version": rubric.VERSION,
        "temperature": 0,
        "enable_thinking": False,
        "dry_run": False,
        "language_model_only": True,
        "enable_cuda_graphs": True,
        "compact_json": True,
        "rubric_version": "single-v1",
    }
    if (
        any(config.get(k) != v for k, v in settings.items())
        or config["packet_sha256"] != packet["packet_sha256"]
        or summary["packet_sha256"] != packet["packet_sha256"]
        or summary["run_config_sha256"] != config_sha
        or summary["dry_run"] is not False
        or config["code"][rubric.__name__] != _sha256(Path(rubric.__file__))
        or config["runner_sha256"] != _sha256(Path(score.__file__))
        or summary["requests_sha256"] != _sha256(scored / "requests.jsonl")
    ):
        raise ValueError("Scoring configuration or request binding changed")
    by_hash = {c["text_sha256"]: c for c in packet["cases"]}
    states, spans, seen = defaultdict(list), [], set()
    provenance = {k: config[k] for k in ("model", "model_revision", "prompt_version")}
    provenance["run_config_sha256"] = config_sha
    with (scored / "requests.jsonl").open() as handle:
        for line in handle:
            task = json.loads(line)
            case = by_hash[task["content_hash"]]
            start, end = task["start"], task["end"]
            expected_kind = "youtube" if case["source"] == "youtube-commons" else "web"
            if (
                type(start) is not int
                or type(end) is not int
                or not 0 <= start < end <= len(case["text"])
                or task["kind"] != expected_kind
                or task["key"] != f"{expected_kind}-{case['text_sha256']}-{start}-{end}"
                or task["key"] in seen
            ):
                raise ValueError("Invalid or repeated span")
            seen.add(task["key"])
            state = engine._cached(
                scored / "decisions" / f"{task['key']}.json",
                task,
                case["text"][start:end],
                rubric.parse_response,
                provenance,
            )
            if state is None:
                raise ValueError("Missing raw response")
            labels = state.get("labels") or {}
            spans.append(
                {
                    **{
                        k: task[k]
                        for k in ("key", "content_hash", "start", "end", "task_sha256")
                    },
                    **provenance,
                    "parse_status": state["parse_status"],
                    **{k: labels.get(k) for k in (*rubric.LABELS, "reason")},
                    "attempts_json": json.dumps(state["attempts"], ensure_ascii=False),
                }
            )
            states[case["text_sha256"]].append(
                (
                    start,
                    end,
                    rubric.route(labels, state["parse_status"]),
                    state["parse_status"],
                )
            )
    if len(seen) != summary["spans"] or set(states) != set(by_hash):
        raise ValueError("Span/text coverage differs")
    decisions = []
    for case in packet["cases"]:
        rows = sorted(states[case["text_sha256"]])
        if (
            rows[0][0] != 0
            or rows[-1][1] != len(case["text"])
            or any(a[1] != b[0] for a, b in zip(rows, rows[1:]))
        ):
            raise ValueError("Spans omit or overlap source characters")
        routes = {r[2] for r in rows}
        decisions.append(
            {
                "case_id": case["case_id"],
                "source": case["source"],
                "content_hash": case["text_sha256"],
                "content_length": case["content_length"],
                "route": next(iter(routes)) if len(routes) == 1 else "review_required",
                "scoring_complete": all(r[3] == "ok" for r in rows),
                "run_config_sha256": config_sha,
                "plan_sha256": _sha256(path),
                "selection_decision": None,
            }
        )
    documents = old.source_rows(packet)
    keeps = {d["case_id"] for d in decisions if d["route"] == "eligible"}
    candidates = [d for d in documents if old.case_id(d) in keeps]
    exports = dict(zip(FILES, (documents, candidates, decisions, spans)))
    if verify_export:
        for name, rows in exports.items():
            if read_rows(directory / name) != rows:
                raise ValueError(f"Export disagrees with source/raw responses: {name}")
    counts = {
        "records": len(documents),
        "input_tokens": sum(d["content_length"] for d in documents),
        "candidate_records": len(candidates),
        "candidate_tokens": sum(d["content_length"] for d in candidates),
        "unresolved_documents": sum(not d["scoring_complete"] for d in decisions),
        "routes": dict(Counter(d["route"] for d in decisions)),
        "spans": len(spans),
        "token_routes": {
            r: sum(d["content_length"] for d in decisions if d["route"] == r)
            for r in ("eligible", "exclude", "review_required")
        },
    }
    return counts, exports


def run_slot(path, plan, slot, factory):
    directory = folder(plan, slot)
    directory.mkdir(parents=True, exist_ok=True)
    with _directory_lock(directory):
        saved = checked_summary(path, plan, slot)
        if saved and saved.get("complete"):
            return 0
        started = time.monotonic()
        _write_json(
            directory / "summary.json",
            {
                "complete": False,
                "workflow_finished": False,
                "plan_sha256": _sha256(path),
            },
        )
        part = plan["parts"][slot]
        packet = score_partitions.part_packet(
            Path(plan["input_dir"]),
            {k: v for k, v in part.items() if k != "index"},
            plan["manifest_sha256"],
        )
        work = directory / "work"
        work.mkdir(exist_ok=True)
        packet_path = work / "packet.json"
        if packet_path.exists() and json.loads(packet_path.read_text()) != packet:
            raise ValueError("Input packet changed")
        _write_json(packet_path, packet)
        args = [
            "--packet",
            str(packet_path),
            "--output-dir",
            str(work / "score"),
            "--model",
            plan["model"]["model"],
            "--model-revision",
            plan["model"]["revision"],
            "--language-model-only",
            "--local-files-only",
            "--enable-cuda-graphs",
            "--compact-json",
        ]
        for key, value in plan["settings"].items():
            args += ["--" + key.replace("_", "-"), str(value)]
        generations = []
        for _ in range(
            2
        ):  # Retry malformed responses once; valid responses are cached.
            status = score.main(args, model_factory=factory)
            generations.append(
                json.loads((work / "score/summary.json").read_text())[
                    "generation_this_invocation"
                ]
            )
            if status == 0:
                break
            if status != 2:
                raise RuntimeError("Unexpected scorer failure")
        counts, exports = audit_part(path, plan, slot, verify_export=False)
        for name, schema in zip(
            FILES, (old.prepare.SCHEMA, old.prepare.SCHEMA, DECISIONS, SPANS)
        ):
            old.atomic_parquet(directory / name, exports[name], schema)
        _write_json(
            directory / "summary.json",
            {
                "complete": counts["unresolved_documents"] == 0,
                "workflow_finished": True,
                "slot": slot,
                "source": part["path"].split("/")[1],
                "plan_sha256": _sha256(path),
                **counts,
                "generation_this_invocation": generations,
                "wall_seconds_this_invocation": time.monotonic() - started,
                "files": {name: _sha256(directory / name) for name in FILES},
                "selection_decision": None,
            },
        )
        return 0 if counts["unresolved_documents"] == 0 else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("plan")
    create.add_argument("--input-dir", type=Path, required=True)
    create.add_argument("--output-dir", type=Path, required=True)
    for name in ("work", "pending"):
        child = sub.add_parser(name)
        child.add_argument("--plan", type=Path, required=True)
        if name == "work":
            child.add_argument("--work-list", type=Path, required=True)
            child.add_argument("--job-index", type=int, required=True)
        else:
            child.add_argument(
                "--source", choices=("benchmark", *SOURCES, "all"), required=True
            )
            child.add_argument("--work-list", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.command == "plan":
        plan = make_plan(args.input_dir, args.output_dir)
        print(
            f"Frozen {len(plan['parts'])} parts; only explicitly submitted work will run."
        )
        return 0
    plan = load_plan(args.plan)
    if args.command == "pending":
        slots = [
            i
            for i, p in enumerate(plan["parts"])
            if (
                args.source == "all"
                or (args.source == "benchmark" and i < 8)
                or p["path"].split("/")[1] == args.source
            )
            and not (checked_summary(args.plan, plan, i) or {}).get("complete")
        ]
        size = 1 if args.source == "benchmark" else 8
        jobs = [slots[i : i + size] for i in range(0, len(slots), size)]
        value = {"plan_sha256": _sha256(args.plan), "jobs": jobs}
        if args.work_list.exists():
            raise ValueError("Refuse to overwrite a submitted work list")
        _write_json(args.work_list, value)
        print(len(jobs))
        return 0
    work = json.loads(args.work_list.read_text())
    if work["plan_sha256"] != _sha256(args.plan) or not 0 <= args.job_index < len(
        work["jobs"]
    ):
        raise ValueError("Work allocation does not match plan")
    slots = work["jobs"][args.job_index]
    if len(slots) != len(set(slots)):
        raise ValueError("Repeated work slot")
    pending = [
        i
        for i in slots
        if not (checked_summary(args.plan, plan, i) or {}).get("complete")
    ]
    if not pending:
        return 0
    cuda_preflight.configure_and_check()
    cuda_preflight.check_sampler()
    factory = runtime.ResidentModel()
    results = [run_slot(args.plan, plan, slot, factory) for slot in pending]
    return max(results, default=0)


if __name__ == "__main__":
    raise SystemExit(main())

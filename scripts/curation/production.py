"""Bounded production scoring, candidate export and resumable work allocation.

Both passes cover every selected document. Existing policy is unchanged:
agreement on eligibility produces a candidate; disagreements remain review items.
Candidate files preserve the prepared schema and are not public release files.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import time

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.youtube.download import _directory_lock, _write_json
from scripts.youtube.profile import _sha256
from scripts.youtube_filter import score as engine
from . import (
    critic_v2,
    cuda_preflight,
    evaluate,
    first_batch,
    prepare,
    rubric_v4,
    runtime,
    score,
    score_partitions,
)

VERSION = "curation-production-v1"
STAGES = {"screen": ("v4", rubric_v4), "critic": ("critic-v2", critic_v2)}
SOURCES = ("primus-fineweb", "youtube-commons")
LOG = logging.getLogger(__name__)
DECISION_SCHEMA = pa.schema(
    [
        ("case_id", pa.string()),
        ("source", pa.string()),
        ("content_hash", pa.string()),
        ("content_length", pa.int64()),
        ("screen_route", pa.string()),
        ("critic_route", pa.string()),
        ("provisional_route", pa.string()),
        ("scoring_complete", pa.bool_()),
        ("screen_config_sha256", pa.string()),
        ("critic_config_sha256", pa.string()),
        ("plan_sha256", pa.string()),
        ("selection_decision", pa.bool_()),
    ]
)
SPAN_SCHEMA = pa.schema(
    [
        ("stage", pa.string()),
        ("key", pa.string()),
        ("kind", pa.string()),
        ("content_hash", pa.string()),
        ("start", pa.int64()),
        ("end", pa.int64()),
        ("task_sha256", pa.string()),
        ("run_config_sha256", pa.string()),
        ("model", pa.string()),
        ("model_revision", pa.string()),
        ("prompt_version", pa.string()),
        ("parse_status", pa.string()),
        *[
            (k, pa.string())
            for k in (
                "security_relevance",
                "technical_substance",
                "text_usability",
                "text_language",
                "mixed_content",
            )
        ],
        ("quality_concerns", pa.list_(pa.string())),
        ("attempts_json", pa.large_string()),
    ]
)


def code_hashes():
    # Bind transitive rubric/parser dependencies and launch/audit code as well.
    paths = sorted(first_batch.HERE.glob("*.py"))
    paths += [
        first_batch.HERE / name
        for name in (
            "run_first_batch.sh",
            "submit_production.sh",
            "production.sbatch",
            "production_audit.sbatch",
            "requirements-next.txt",
            "next_models.json",
        )
    ]
    paths += [
        Path(engine.__file__),
        first_batch.HERE.parent / "youtube_filter/rubric.py",
        first_batch.HERE.parent / "youtube_filter/rubric_segments.py",
        first_batch.HERE.parent.parent / "src/ingest/utils.py",
    ]
    return {
        str(p.relative_to(first_batch.HERE.parent.parent)): _sha256(p) for p in paths
    }


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def choose_parts(summary, count):
    if count < 1:
        raise ValueError("Partition allocation must be positive")
    # Avoid the original diagnostic source shards, not merely their selected parts.
    old_shards = {
        str(Path(p["path"]).parent) for p in first_batch.select_parts(summary)
    }
    selected = []
    for source in SOURCES:
        choices = [
            (i, p)
            for i, p in enumerate(summary["parts"])
            if p["path"].split("/")[1] == source
            and str(Path(p["path"]).parent) not in old_shards
        ]
        choices.sort(
            key=lambda x: hashlib.sha256(
                f"{VERSION}:{x[1]['path']}".encode()
            ).hexdigest()
        )
        seen = set()
        for i, part in choices:
            shard = str(Path(part["path"]).parent)
            if shard in seen:
                continue
            seen.add(shard)
            selected.append({"index": i, **part})
            if len(seen) == count:
                break
        if len(seen) != count:
            raise ValueError(f"Not enough fresh source shards for {source}")
    # Start both sources promptly when Slurm schedules slots in array order.
    return [
        selected[source * count + i]
        for i in range(count)
        for source in range(len(SOURCES))
    ]


def make_plan(root, output, count=4, audit_per_stratum=20):
    root, output = root.resolve(), output.resolve()
    if root == output or root.is_relative_to(output) or output.is_relative_to(root):
        raise ValueError("Production output must be separate from prepared input")
    if audit_per_stratum < 1:
        raise ValueError("Audit workload must be positive")
    prepared = score_partitions.load_prepared(root)
    plan = {
        "version": VERSION,
        "input_dir": str(root),
        "output_dir": str(output),
        "manifest_sha256": _sha256(root / "summary.json"),
        "lineage_sha256": prepared["index"]["lineage_sha256"],
        "parts": choose_parts(prepared, count),
        "model": json.loads((first_batch.HERE / "next_models.json").read_text())[
            "models"
        ][1],
        "code": code_hashes(),
        "audit_per_stratum": audit_per_stratum,
        "settings": {
            "tensor_parallel_size": 1,
            "batch_size": 32,
            "max_model_len": 8192,
            "focal_tokens": 4096,
            "max_output_tokens": 1536,
            "compact_json": True,
        },
        "scope": "Initial production batch; provisional candidates only. No final selection, cross-source deduplication or publication approval. Partition and audit counts allocate work, not quality thresholds.",
    }
    output.mkdir(parents=True, exist_ok=True)
    with _directory_lock(output):
        path = output / "plan.json"
        if path.exists() and json.loads(path.read_text()) != plan:
            raise ValueError("Production plan changed; use a new output directory")
        _write_json(path, plan)
    return plan


def load_plan(path):
    plan = json.loads(path.read_text())
    if (
        plan["version"] != VERSION
        or plan["code"] != code_hashes()
        or path.resolve() != Path(plan["output_dir"]) / "plan.json"
    ):
        raise ValueError("Production code or output binding changed")
    root = Path(plan["input_dir"])
    if _sha256(root / "summary.json") != plan["manifest_sha256"]:
        raise ValueError("Prepared manifest changed")
    summary = score_partitions.load_prepared(root)
    indices = [p["index"] for p in plan["parts"]]
    if len(indices) != len(set(indices)):
        raise ValueError("Repeated production partition")
    for part in plan["parts"]:
        if summary["parts"][part["index"]] != {
            k: v for k, v in part.items() if k != "index"
        }:
            raise ValueError("Production partition differs from manifest")
    return plan


def part_dir(plan, slot):
    if not 0 <= slot < len(plan["parts"]):
        raise ValueError("Slot is outside the frozen plan")
    return Path(plan["output_dir"]) / "parts" / f"{slot:04d}"


def checked_summary(plan_path, plan, slot):
    folder = part_dir(plan, slot)
    path = folder / "summary.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text())
    if summary["plan_sha256"] != _sha256(plan_path):
        raise ValueError("Partition checkpoint belongs to another plan")
    if summary.get("workflow_finished"):
        if set(summary["files"]) != {
            "documents.parquet",
            "candidates.parquet",
            "decisions.parquet",
            "spans.parquet",
        }:
            raise ValueError("Incomplete production file manifest")
        for name, checksum in summary["files"].items():
            if _sha256(folder / name) != checksum:
                raise ValueError(
                    f"Production output checksum mismatch: {folder / name}"
                )
    return summary


def atomic_parquet(path, rows, schema):
    temporary = path.with_suffix(".tmp")
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd"
    )
    temporary.replace(path)


def source_rows(packet):
    return [
        {
            **c["lineage"],
            "source": c["source"],
            "kind": "youtube" if c["source"] == "youtube-commons" else "web",
            "text": c["text"],
            "content_hash": c["text_sha256"],
            "content_length": c["content_length"],
        }
        for c in packet["cases"]
    ]


def case_id(row):
    return f"{row['source']}:{row['source_shard']}:{row['source_row']}"


def span_rows(packet, work, stage):
    _, rubric = STAGES[stage]
    run = work / stage
    config = json.loads((run / "run-config.json").read_text())
    provenance = {k: config[k] for k in ("model", "model_revision", "prompt_version")}
    provenance["run_config_sha256"] = _sha256(run / "run-config.json")
    texts = {c["text_sha256"]: c["text"] for c in packet["cases"]}
    rows = []
    with (run / "requests.jsonl").open() as f:
        for line in f:
            t = json.loads(line)
            state = engine._cached(
                run / "decisions" / f"{t['key']}.json",
                t,
                texts[t["content_hash"]][t["start"] : t["end"]],
                rubric.parse_response,
                provenance,
            )
            if state is None:
                raise ValueError("Finished run is missing a raw checkpoint")
            labels = state.get("labels") or {}
            rows.append(
                {
                    "stage": stage,
                    **{
                        k: t[k]
                        for k in (
                            "key",
                            "kind",
                            "content_hash",
                            "start",
                            "end",
                            "task_sha256",
                        )
                    },
                    **provenance,
                    "parse_status": state["parse_status"],
                    **{
                        k: labels.get(k)
                        for k in (
                            "security_relevance",
                            "technical_substance",
                            "text_usability",
                            "text_language",
                            "mixed_content",
                        )
                    },
                    "quality_concerns": [
                        c["kind"] for c in labels.get("quality_concerns", [])
                    ],
                    "attempts_json": json.dumps(state["attempts"], ensure_ascii=False),
                }
            )
    return rows


def combine(packet, reports, plan_sha):
    second = {r["case_id"]: r for r in reports["critic"]["comparisons"]}
    cases = {c["case_id"]: c for c in packet["cases"]}
    rows = []
    for a in reports["screen"]["comparisons"]:
        b = second[a["case_id"]]
        complete = all(s == "ok" for r in (a, b) for s in r["parse_statuses"])
        route = (
            a["model_route"]
            if complete and a["model_route"] == b["model_route"]
            else "review_required"
        )
        rows.append(
            {
                "case_id": a["case_id"],
                "source": a["source"],
                "content_hash": cases[a["case_id"]]["text_sha256"],
                "content_length": a["content_length"],
                "screen_route": a["model_route"],
                "critic_route": b["model_route"],
                "provisional_route": route,
                "scoring_complete": complete,
                "screen_config_sha256": reports["screen"]["run_config_sha256"],
                "critic_config_sha256": reports["critic"]["run_config_sha256"],
                "plan_sha256": plan_sha,
                "selection_decision": None,
            }
        )
    if set(second) != set(cases) or {r["case_id"] for r in rows} != set(cases):
        raise ValueError("Scoring coverage differs from input")
    return rows


def run_slot(plan_path, slot):
    plan = load_plan(plan_path)
    folder = part_dir(plan, slot)
    folder.mkdir(parents=True, exist_ok=True)
    with _directory_lock(folder):
        old = checked_summary(plan_path, plan, slot)
        if old and old.get("complete"):
            LOG.info("Slot %s already complete; no model loaded", slot)
            return 0
        plan_sha = _sha256(plan_path)
        _write_json(
            folder / "summary.json",
            {"complete": False, "workflow_finished": False, "plan_sha256": plan_sha},
        )
        part = plan["parts"][slot]
        packet = score_partitions.part_packet(
            Path(plan["input_dir"]),
            {k: v for k, v in part.items() if k != "index"},
            plan["manifest_sha256"],
        )
        work = folder / "work"
        work.mkdir(exist_ok=True)
        packet_path = work / "packet.json"
        if packet_path.exists() and json.loads(packet_path.read_text()) != packet:
            raise ValueError("Production input packet changed")
        _write_json(packet_path, packet)
        # Recompute source token counts/hashes before any GPU work.
        score.load_packet(packet_path)
        environment = {
            "cuda_compiler": cuda_preflight.configure_and_check(),
            "sampler": cuda_preflight.check_sampler(),
        }
        env_path = work / "runtime.json"
        if env_path.exists() and json.loads(env_path.read_text()) != environment:
            raise ValueError(
                "GPU runtime changed; preserve this run and use a new output directory"
            )
        _write_json(env_path, environment)
        factory = runtime.ResidentModel()
        reports, spans, stage_seconds = {}, [], {}
        for stage, (rubric_version, _) in STAGES.items():
            args = [
                "--packet",
                str(packet_path),
                "--output-dir",
                str(work / stage),
                "--rubric-version",
                rubric_version,
                "--model",
                plan["model"]["model"],
                "--model-revision",
                plan["model"]["revision"],
                "--language-model-only",
                "--local-files-only",
                "--enable-cuda-graphs",
                "--compact-json",
            ]
            for k, v in plan["settings"].items():
                if k != "compact_json":
                    args += ["--" + k.replace("_", "-"), str(v)]
            # One automatic retry of unresolved outputs; successful checkpoints are reused.
            started = time.monotonic()
            status = score.main(args, model_factory=factory)
            if status == 2:
                status = score.main(args, model_factory=factory)
            if status not in (0, 2):
                raise RuntimeError(f"Scoring {stage} exited with {status}")
            stage_seconds[stage] = time.monotonic() - started
            reports[stage] = evaluate.evaluate(packet_path, score_dir=work / stage)
            spans.extend(span_rows(packet, work, stage))
        decisions = combine(packet, reports, plan_sha)
        documents = source_rows(packet)
        eligible = {
            r["case_id"] for r in decisions if r["provisional_route"] == "eligible"
        }
        candidates = [r for r in documents if case_id(r) in eligible]
        atomic_parquet(folder / "documents.parquet", documents, prepare.SCHEMA)
        atomic_parquet(folder / "candidates.parquet", candidates, prepare.SCHEMA)
        atomic_parquet(folder / "decisions.parquet", decisions, DECISION_SCHEMA)
        atomic_parquet(folder / "spans.parquet", spans, SPAN_SCHEMA)
        from .production_audit import audit_part

        audited = audit_part(folder, plan, plan_sha, slot)
        complete = audited["unresolved_documents"] == 0
        _write_json(
            folder / "summary.json",
            {
                "complete": complete,
                "workflow_finished": True,
                "plan_sha256": plan_sha,
                "slot": slot,
                "part_index": part["index"],
                **audited,
                "files": {
                    n: _sha256(folder / n)
                    for n in (
                        "documents.parquet",
                        "candidates.parquet",
                        "decisions.parquet",
                        "spans.parquet",
                    )
                },
                "generation_latest_call": {
                    s: r["generation"] for s, r in reports.items()
                },
                "stage_wall_seconds_this_invocation": stage_seconds,
                "selection_decision": None,
                "scope": plan["scope"],
            },
        )
        return 0 if complete else 2


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    create = sub.add_parser("plan")
    create.add_argument("--input-dir", type=Path, required=True)
    create.add_argument("--output-dir", type=Path, required=True)
    create.add_argument("--parts-per-source", type=int, default=4)
    create.add_argument("--audit-per-stratum", type=int, default=20)
    for name in ("run", "pending"):
        child = sub.add_parser(name)
        child.add_argument("--plan", type=Path, required=True)
        if name == "run":
            child.add_argument("--slot", type=int, required=True)
    a = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if a.command == "plan":
        plan = make_plan(
            a.input_dir, a.output_dir, a.parts_per_source, a.audit_per_stratum
        )
        print(
            json.dumps(
                {
                    "slots": len(plan["parts"]),
                    "records": sum(x["records"] for x in plan["parts"]),
                    "candidate_input_tokens": sum(x["tokens"] for x in plan["parts"]),
                    "parts": plan["parts"],
                },
                indent=2,
            )
        )
        return 0
    if a.command == "run":
        return run_slot(a.plan, a.slot)
    plan = load_plan(a.plan)
    print(
        ",".join(
            str(i)
            for i in range(len(plan["parts"]))
            if not (checked_summary(a.plan, plan, i) or {}).get("complete")
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

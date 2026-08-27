#!/usr/bin/env python3
"""Filter FineWeb for security-relevant documents and write normalized Parquet."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != SCRIPT_DIR]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ingest.connectors.web import (
    DsirScorer,
    audit_fineweb_output,
    build_slurm_script,
    docs_from_input,
    fineweb_record_text,
    normalize_fineweb_record,
    write_fineweb_records,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.audit:
        summary = audit_fineweb_output(args.output_dir, args.report_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.slurm:
        return _submit_slurm(args)
    return _run_filter(args)


def _run_filter(args: argparse.Namespace) -> int:
    scorer = DsirScorer.load(args.scorer)
    shard_name = args.shard_name or ("local" if args.task_id is None else f"task_{args.task_id:05d}")
    chunked = args.flush_every > 0
    state_path = args.report_dir / f"{shard_name}_state.json"
    output_path = args.output_dir / "source_id=fineweb-security" / f"{shard_name}.parquet"
    if not chunked and output_path.exists() and not args.overwrite:
        print(f"Output exists, skipping: {output_path}", flush=True)
        return 0
    state = _load_state(state_path) if chunked and not args.overwrite else {}
    if state.get("completed"):
        print(f"Task already completed, skipping: {state_path}", flush=True)
        return 0
    if chunked and args.overwrite:
        _remove_existing_chunks(args.output_dir, shard_name)

    kept = []
    score_rows: list[dict[str, Any]] = []
    score_values: list[float] = []
    seen = int(state.get("seen", 0))
    scored = int(state.get("scored", 0))
    rejected = int(state.get("rejected", 0))
    written = int(state.get("kept", 0))
    resume_after = int(state.get("last_global_index", -1))
    chunk_index = int(state.get("next_chunk", 0)) if chunked else 0
    if chunked and chunk_index == 0 and not args.overwrite:
        chunk_index = _next_chunk_index(args.output_dir, shard_name)
    start_time = time.monotonic()
    deadline = start_time + args.time_budget_minutes * 60 if args.time_budget_minutes and args.time_budget_minutes > 0 else None
    stopped_for_time = False
    last_global_index = resume_after
    print(
        json.dumps(
            {
                "event": "fineweb_task_start",
                "task_id": args.task_id,
                "tasks": args.tasks,
                "shard_name": shard_name,
                "resume_after": resume_after,
                "chunked": chunked,
                "flush_every": args.flush_every,
                "time_budget_minutes": args.time_budget_minutes,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for global_index, record in enumerate(
        docs_from_input(
            input_glob=args.input_glob,
            fineweb_dataset=args.fineweb_dataset,
            fineweb_config=args.fineweb_config,
            split=args.split,
        )
    ):
        if global_index <= resume_after:
            continue
        if args.task_id is not None and global_index % args.tasks != args.task_id:
            continue
        if args.max_docs is not None and scored >= args.max_docs:
            break
        last_global_index = global_index
        seen += 1
        text = fineweb_record_text(record)
        if not text:
            rejected += 1
            continue
        token_estimate = max(1, len(text.split()))
        if token_estimate < args.min_words or token_estimate > args.max_words:
            rejected += 1
            continue
        score = scorer.score(text)
        score_values.append(score)
        scored += 1
        keep = score >= args.min_score
        if keep:
            try:
                kept.append(normalize_fineweb_record(record, score=score))
            except Exception:
                rejected += 1
                continue
        else:
            rejected += 1
        if chunked and len(kept) >= args.flush_every:
            chunk_written = _flush_chunk(
                kept,
                args.output_dir,
                shard_name=shard_name,
                chunk_index=chunk_index,
                overwrite=args.overwrite,
            )
            written += chunk_written
            kept.clear()
            chunk_index += 1
            _write_state(
                state_path,
                completed=False,
                task_id=args.task_id,
                tasks=args.tasks,
                seen=seen,
                scored=scored,
                kept=written,
                rejected=rejected,
                last_global_index=last_global_index,
                next_chunk=chunk_index,
            )
        if len(score_rows) < args.audit_sample_size:
            score_rows.append(
                {
                    "global_index": global_index,
                    "score": score,
                    "kept": keep,
                    "title": _field(record, "title"),
                    "url": _field(record, "url") or _field(record, "source_url"),
                    "preview": text[:300].replace("\n", " "),
                }
            )
        if args.progress_every and scored and scored % args.progress_every == 0:
            _print_progress(shard_name, seen=seen, scored=scored, kept=written + len(kept), rejected=rejected)
        if deadline is not None and time.monotonic() >= deadline:
            stopped_for_time = True
            print(f"Time budget reached for {shard_name}; flushing and exiting cleanly.", flush=True)
            break

    if chunked:
        if kept:
            written += _flush_chunk(
                kept,
                args.output_dir,
                shard_name=shard_name,
                chunk_index=chunk_index,
                overwrite=args.overwrite,
            )
            kept.clear()
            chunk_index += 1
        _write_state(
            state_path,
            completed=not stopped_for_time and (args.max_docs is None or scored >= args.max_docs),
            task_id=args.task_id,
            tasks=args.tasks,
            seen=seen,
            scored=scored,
            kept=written,
            rejected=rejected,
            last_global_index=last_global_index,
            next_chunk=chunk_index,
        )
    else:
        written = write_fineweb_records(kept, args.output_dir, shard_name=shard_name, overwrite=args.overwrite)
    metrics = {
        "task_id": args.task_id,
        "tasks": args.tasks,
        "seen": seen,
        "scored": scored,
        "kept": written,
        "rejected": rejected,
        "min_score": args.min_score,
        "output": str(output_path) if not chunked else str(args.output_dir / "source_id=fineweb-security" / f"{shard_name}_part_*.parquet"),
        "last_global_index": last_global_index,
        "stopped_for_time_budget": stopped_for_time,
        "score_distribution": _distribution(score_values),
    }
    _write_task_reports(args.report_dir, shard_name, metrics, score_rows, args.audit_sample_size)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    return 0


def _submit_slurm(args: argparse.Namespace) -> int:
    args.log_dir.mkdir(parents=True, exist_ok=True)
    command = _single_task_command(args)
    script = build_slurm_script(
        tasks=args.tasks,
        array_concurrency=args.array_concurrency,
        account=args.slurm_account,
        partition=args.slurm_partition,
        qos=args.slurm_qos,
        cpus_per_task=args.cpus_per_task,
        mem=args.mem,
        time_limit=args.time,
        command=command,
    )
    script_path = args.log_dir / "run_fineweb_ingest.sbatch"
    script_path.write_text(script, encoding="utf-8")
    print(f"Wrote Slurm script: {script_path}")
    if args.dry_run:
        print(script)
        return 0
    proc = subprocess.run(["sbatch", str(script_path)], text=True, check=False)
    return proc.returncode


def _single_task_command(args: argparse.Namespace) -> str:
    bits = [
        "python3",
        "scripts/ingest_fineweb.py",
        "--scorer",
        str(args.scorer),
        "--output-dir",
        str(args.output_dir),
        "--report-dir",
        str(args.report_dir),
        "--min-score",
        str(args.min_score),
    ]
    if args.input_glob:
        bits.extend(["--input-glob", args.input_glob])
    else:
        bits.extend(["--fineweb-dataset", args.fineweb_dataset, "--split", args.split])
        if args.fineweb_config:
            bits.extend(["--fineweb-config", args.fineweb_config])
    if args.max_docs is not None:
        bits.extend(["--max-docs", str(args.max_docs)])
    bits.extend(["--flush-every", str(args.flush_every)])
    bits.extend(["--time-budget-minutes", str(args.time_budget_minutes)])
    bits.extend(["--progress-every", str(args.progress_every)])
    if args.overwrite:
        bits.append("--overwrite")
    return " ".join(shlex.quote(bit) for bit in bits)


def _flush_chunk(records: list, output_dir: Path, *, shard_name: str, chunk_index: int, overwrite: bool) -> int:
    chunk_name = f"{shard_name}_part_{chunk_index:05d}"
    written = write_fineweb_records(records, output_dir, shard_name=chunk_name, overwrite=overwrite)
    print(f"Wrote {written:,} kept records to {chunk_name}.parquet", flush=True)
    return written


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_state(path: Path, **state: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at_unix"] = time.time()
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _next_chunk_index(output_dir: Path, shard_name: str) -> int:
    source_dir = output_dir / "source_id=fineweb-security"
    indices: list[int] = []
    for path in source_dir.glob(f"{shard_name}_part_*.parquet"):
        try:
            indices.append(int(path.stem.rsplit("_part_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return (max(indices) + 1) if indices else 0


def _remove_existing_chunks(output_dir: Path, shard_name: str) -> None:
    source_dir = output_dir / "source_id=fineweb-security"
    for path in source_dir.glob(f"{shard_name}_part_*.parquet"):
        path.unlink()


def _print_progress(shard_name: str, *, seen: int, scored: int, kept: int, rejected: int) -> None:
    print(
        json.dumps(
            {
                "event": "fineweb_progress",
                "shard_name": shard_name,
                "seen": seen,
                "scored": scored,
                "kept": kept,
                "rejected": rejected,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _write_task_reports(report_dir: Path, shard_name: str, metrics: dict[str, Any], rows: list[dict[str, Any]], sample_size: int) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{shard_name}_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    rows = sorted(rows, key=lambda row: row["score"], reverse=True)[:sample_size]
    with (report_dir / f"{shard_name}_audit_sample.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["global_index", "score", "kept", "title", "url", "preview"])
        writer.writeheader()
        writer.writerows(rows)


def _field(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    return value if isinstance(value, str) else ""


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "p50": None, "p90": None, "p99": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(values) / len(values),
        "p50": ordered[round((len(ordered) - 1) * 0.50)],
        "p90": ordered[round((len(ordered) - 1) * 0.90)],
        "p99": ordered[round((len(ordered) - 1) * 0.99)],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer", type=Path, default=Path("data/fineweb/dsir_scorer.pkl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/fineweb/normalized"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/fineweb"))
    parser.add_argument("--fineweb-dataset", default="HuggingFaceFW/fineweb")
    parser.add_argument("--fineweb-config")
    parser.add_argument("--split", default="train")
    parser.add_argument("--input-glob")
    parser.add_argument("--max-docs", type=int)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--min-words", type=int, default=100)
    parser.add_argument("--max-words", type=int, default=100_000)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--tasks", type=int, default=1)
    parser.add_argument("--shard-name")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--flush-every", type=int, default=1000, help="Write a Parquet chunk after this many kept docs; use 0 for one final write.")
    parser.add_argument("--time-budget-minutes", type=float, default=230.0, help="Stop cleanly before Slurm wall time; use 0 to disable.")
    parser.add_argument("--progress-every", type=int, default=10_000, help="Print progress after this many scored docs; use 0 to disable.")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--audit-sample-size", type=int, default=40)
    parser.add_argument("--slurm", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--array-concurrency", type=int, default=8)
    parser.add_argument("--slurm-account", default="marlowe-m000091")
    parser.add_argument("--slurm-partition", default="preempt")
    parser.add_argument("--slurm-qos", default="normal")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", default="32G")
    parser.add_argument("--time", default="04:00:00")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/fineweb"))
    return parser


if __name__ == "__main__":
    raise SystemExit(main())

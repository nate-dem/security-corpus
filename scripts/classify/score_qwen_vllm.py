#!/usr/bin/env python3
"""Score records with local Qwen via vLLM.

Dry-run mode renders prompts and never imports vLLM, loads Qwen weights, or uses
GPUs. Real inference is intended for the Slurm GPU cluster and should be run
manually by the researcher.
"""

from __future__ import annotations

import argparse
from fnmatch import fnmatchcase
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import pyarrow.dataset as ds


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) in sys.path:
    sys.path.remove(str(SRC))
sys.path.insert(0, str(SRC))

from classify.io import ensure_parent  # noqa: E402
from classify.qwen import (  # noqa: E402
    QWEN_OPTIONAL_COLUMNS,
    QWEN_PROMPT_VERSIONS,
    QWEN_REQUIRED_COLUMNS,
    QwenTask,
    coerce_task,
    make_qwen_sidecar_row,
    parse_qwen_response,
    qwen_schema_extra_fields,
    render_prompt,
)
from classify.sidecar import qwen_sidecar_schema, write_sidecar_rows  # noqa: E402


DEFAULT_MODEL = "Qwen/Qwen3-4B"


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    task = coerce_task(args.task)
    prompt_version = args.prompt_version or QWEN_PROMPT_VERSIONS[task]

    candidate_keys = _load_candidate_keys(
        args.candidate_sidecar,
        args.candidate_keep_column,
    )
    source_ids, source_like = _resolve_source_filters(args)
    existing_keys = _load_existing_output_keys(args.output_dir) if args.output_dir else set()

    if args.dry_run:
        prompts = _collect_dry_run_prompts(
            args,
            task,
            prompt_version,
            candidate_keys,
            existing_keys,
            source_ids,
            source_like,
        )
        if args.dry_run_prompts:
            _write_jsonl(args.dry_run_prompts, prompts)
            print(f"Wrote {len(prompts)} dry-run prompts to {args.dry_run_prompts}")
        else:
            print(json.dumps(prompts[: args.preview_limit], indent=2))
        print("Dry run complete; vLLM was not imported and no model weights were loaded.")
        return 0

    if args.output_dir is None:
        raise ValueError("--output-dir is required for real Qwen inference")

    _run_vllm_scoring(
        args=args,
        task=task,
        prompt_version=prompt_version,
        candidate_keys=candidate_keys,
        existing_keys=existing_keys,
        source_ids=source_ids,
        source_like=source_like,
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Normalized corpus Parquet file/dir.")
    parser.add_argument(
        "--task",
        choices=[task.value for task in QwenTask] + [task.value.replace("_", "-") for task in QwenTask],
        required=True,
        help="Qwen task to run.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Shard output/cache directory.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-version", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-content-chars", type=int, default=24_000)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--max-model-len", type=int, default=32_768)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--shard-id", default="0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--candidate-sidecar",
        type=Path,
        default=None,
        help="Optional sidecar whose keys define records eligible for scoring.",
    )
    parser.add_argument(
        "--candidate-keep-column",
        default=None,
        help="Optional boolean column in --candidate-sidecar; only true rows are loaded.",
    )
    parser.add_argument(
        "--parse-failure-should-keep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fallback should_keep value for parse failures. Defaults to keep-for-review.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--source-id",
        action="append",
        default=[],
        help="Exact source_id to score. Repeat for multiple sources.",
    )
    parser.add_argument(
        "--source-like",
        action="append",
        default=[],
        help="Glob-style source_id pattern to score, e.g. 'reddit-*'. Repeatable.",
    )
    parser.add_argument(
        "--qa-sources",
        action="store_true",
        help="Score only QA/social sources: stackoverflow, stackexchange-*, reddit-*.",
    )
    parser.add_argument(
        "--dry-run-prompts",
        type=Path,
        default=None,
        help="Optional JSONL path for rendered prompts in dry-run mode.",
    )
    parser.add_argument("--preview-limit", type=int, default=3)
    return parser


def _collect_dry_run_prompts(
    args: argparse.Namespace,
    task: QwenTask,
    prompt_version: str,
    candidate_keys: set[tuple[str, str, str]] | None,
    existing_keys: set[tuple[str, str, str]],
    source_ids: set[str],
    source_like: list[str],
) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for row in _iter_input_rows(
        args,
        task,
        candidate_keys,
        existing_keys,
        source_ids,
        source_like,
    ):
        prompt = render_prompt(row, task, max_content_chars=args.max_content_chars)
        prompts.append(
            {
                "source_id": row.get("source_id"),
                "record_id": row.get("record_id"),
                "content_hash": row.get("content_hash"),
                "task": task.value,
                "prompt_version": prompt_version,
                "prompt": prompt,
            }
        )
        if args.max_records is not None and len(prompts) >= args.max_records:
            break
    return prompts


def _run_vllm_scoring(
    *,
    args: argparse.Namespace,
    task: QwenTask,
    prompt_version: str,
    candidate_keys: set[tuple[str, str, str]] | None,
    existing_keys: set[tuple[str, str, str]],
    source_ids: set[str],
    source_like: list[str],
) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    schema = qwen_sidecar_schema(qwen_schema_extra_fields(task))
    batch_rows: list[dict[str, Any]] = []
    batch_prompts: list[str] = []
    scored = 0
    part_index = _next_part_index(args.output_dir, args.shard_id)

    for row in _iter_input_rows(
        args,
        task,
        candidate_keys,
        existing_keys,
        source_ids,
        source_like,
    ):
        batch_rows.append(row)
        batch_prompts.append(
            render_prompt(
                row,
                task,
                tokenizer=tokenizer,
                max_content_chars=args.max_content_chars,
            )
        )
        if len(batch_rows) >= args.batch_size:
            part_index, scored = _score_and_write_batch(
                llm=llm,
                sampling_params=sampling_params,
                rows=batch_rows,
                prompts=batch_prompts,
                schema=schema,
                output_dir=args.output_dir,
                task=task,
                model=args.model,
                prompt_version=prompt_version,
                shard_id=args.shard_id,
                part_index=part_index,
                scored=scored,
                parse_failure_should_keep=args.parse_failure_should_keep,
            )
            batch_rows = []
            batch_prompts = []
        if args.max_records is not None and scored + len(batch_rows) >= args.max_records:
            break

    if batch_rows:
        part_index, scored = _score_and_write_batch(
            llm=llm,
            sampling_params=sampling_params,
            rows=batch_rows,
            prompts=batch_prompts,
            schema=schema,
            output_dir=args.output_dir,
            task=task,
            model=args.model,
            prompt_version=prompt_version,
            shard_id=args.shard_id,
            part_index=part_index,
            scored=scored,
            parse_failure_should_keep=args.parse_failure_should_keep,
        )

    print(f"Scored {scored} records into {args.output_dir}")


def _score_and_write_batch(
    *,
    llm: Any,
    sampling_params: Any,
    rows: list[dict[str, Any]],
    prompts: list[str],
    schema: Any,
    output_dir: Path,
    task: QwenTask,
    model: str,
    prompt_version: str,
    shard_id: str,
    part_index: int,
    scored: int,
    parse_failure_should_keep: bool | None,
) -> tuple[int, int]:
    outputs = llm.generate(prompts, sampling_params)
    sidecar_rows = []
    for row, output in zip(rows, outputs):
        response_text = output.outputs[0].text if output.outputs else ""
        parsed = parse_qwen_response(
            response_text,
            parse_failure_should_keep=parse_failure_should_keep,
        )
        sidecar_rows.append(
            make_qwen_sidecar_row(
                row,
                parsed,
                task=task,
                model=model,
                prompt_version=prompt_version,
                shard_id=shard_id,
                raw_response=response_text,
            )
        )

    output_path = output_dir / f"part-shard-{shard_id}-{part_index:06d}.parquet"
    write_sidecar_rows(output_path, sidecar_rows, schema)
    return part_index + 1, scored + len(rows)


def _iter_input_rows(
    args: argparse.Namespace,
    task: QwenTask,
    candidate_keys: set[tuple[str, str, str]] | None,
    existing_keys: set[tuple[str, str, str]],
    source_ids: set[str],
    source_like: list[str],
) -> Iterable[dict[str, Any]]:
    dataset = _open_filtered_dataset(args.input, source_ids, source_like)
    columns = _available_task_columns(dataset, task)
    scanner = dataset.scanner(columns=columns, batch_size=max(args.batch_size, 1))
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    shard_id = int(args.shard_id)
    if shard_id < 0 or shard_id >= args.num_shards:
        raise ValueError("--shard-id must be in the range [0, --num-shards)")
    emitted = 0
    global_index = 0
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            if not _source_matches(row.get("source_id"), source_ids, source_like):
                continue
            key = _key(row)
            if args.num_shards > 1 and global_index % args.num_shards != shard_id:
                global_index += 1
                continue
            global_index += 1
            if candidate_keys is not None and key not in candidate_keys:
                continue
            if key in existing_keys:
                continue
            _validate_required_values(row, task)
            for column in QWEN_OPTIONAL_COLUMNS[task]:
                row.setdefault(column, None)
            yield row
            emitted += 1
            if args.max_records is not None and emitted >= args.max_records:
                return


def _resolve_source_filters(args: argparse.Namespace) -> tuple[set[str], list[str]]:
    source_ids = {source_id for source_id in args.source_id if source_id}
    source_like = [pattern for pattern in args.source_like if pattern]
    if args.qa_sources:
        source_ids.add("stackoverflow")
        source_like.extend(["stackexchange-*", "reddit-*"])
    return source_ids, source_like


def _source_matches(
    source_id: Any,
    source_ids: set[str],
    source_like: Sequence[str],
) -> bool:
    if not source_ids and not source_like:
        return True
    value = str(source_id or "")
    if value in source_ids:
        return True
    return any(fnmatchcase(value, pattern) for pattern in source_like)


def _open_filtered_dataset(
    input_path: Path,
    source_ids: set[str],
    source_like: Sequence[str],
) -> ds.Dataset:
    filtered_files = _source_filtered_parquet_files(input_path, source_ids, source_like)
    if filtered_files is not None:
        return ds.dataset([str(path) for path in filtered_files], format="parquet")
    return ds.dataset(str(input_path), format="parquet", partitioning="hive")


def _source_filtered_parquet_files(
    input_path: Path,
    source_ids: set[str],
    source_like: Sequence[str],
) -> list[Path] | None:
    if not source_ids and not source_like:
        return None
    if not input_path.is_dir():
        return None

    partition_dirs = _source_partition_dirs(input_path)
    if not partition_dirs:
        return None

    matching_dirs = [
        path
        for source_id, path in partition_dirs
        if _source_matches(source_id, source_ids, source_like)
    ]
    if not matching_dirs:
        filters = _format_source_filters(source_ids, source_like)
        raise FileNotFoundError(
            f"No matching source_id partitions found under {input_path} for {filters}"
        )

    parquet_files = sorted(
        {
            file
            for directory in matching_dirs
            for file in directory.glob("**/*.parquet")
            if not file.name.startswith("._")
        }
    )
    if not parquet_files:
        filters = _format_source_filters(source_ids, source_like)
        raise FileNotFoundError(
            f"Matching source_id partitions under {input_path} contain no Parquet files for {filters}"
        )
    return parquet_files


def _source_partition_dirs(input_path: Path) -> list[tuple[str, Path]]:
    candidates: set[Path] = set()
    if input_path.name.startswith("source_id="):
        candidates.add(input_path)
    candidates.update(
        path
        for path in input_path.glob("**/source_id=*")
        if path.is_dir()
    )

    partitions: list[tuple[str, Path]] = []
    for path in sorted(candidates):
        source_id = path.name.removeprefix("source_id=")
        if source_id:
            partitions.append((source_id, path))
    return partitions


def _format_source_filters(source_ids: set[str], source_like: Sequence[str]) -> str:
    parts = []
    if source_ids:
        parts.append("source_id=" + ",".join(sorted(source_ids)))
    if source_like:
        parts.append("source_like=" + ",".join(source_like))
    return "; ".join(parts) if parts else "no source filters"


def _available_task_columns(dataset: ds.Dataset, task: QwenTask) -> list[str]:
    available = set(dataset.schema.names)
    required_columns = list(QWEN_REQUIRED_COLUMNS[task])
    missing = [column for column in required_columns if column not in available]
    if missing:
        raise ValueError(
            f"Input is missing required {task.value} prompt columns: {', '.join(missing)}"
        )
    optional_columns = [
        column
        for column in QWEN_OPTIONAL_COLUMNS[task]
        if column in available and column not in required_columns
    ]
    return required_columns + optional_columns


def _validate_required_values(row: Mapping[str, Any], task: QwenTask) -> None:
    missing_or_blank = [
        column
        for column in QWEN_REQUIRED_COLUMNS[task]
        if _is_missing_prompt_value(row.get(column))
    ]
    if missing_or_blank:
        record_id = row.get("record_id") or "<unknown record_id>"
        raise ValueError(
            f"Input record {record_id} has missing/blank required "
            f"{task.value} prompt fields: {', '.join(missing_or_blank)}"
        )


def _is_missing_prompt_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _load_candidate_keys(
    path: Path | None,
    keep_column: str | None,
) -> set[tuple[str, str, str]] | None:
    if path is None:
        return None
    dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
    columns = ["source_id", "record_id", "content_hash"]
    if keep_column is not None:
        if keep_column not in dataset.schema.names:
            raise ValueError(f"Candidate sidecar is missing {keep_column}")
        columns.append(keep_column)
    keys = set()
    for batch in dataset.scanner(columns=columns).to_batches():
        for row in batch.to_pylist():
            if keep_column is not None and not row.get(keep_column):
                continue
            keys.add(_key(row))
    return keys


def _load_existing_output_keys(output_dir: Path | None) -> set[tuple[str, str, str]]:
    if output_dir is None or not output_dir.exists():
        return set()
    files = sorted(output_dir.glob("*.parquet"))
    if not files:
        return set()
    dataset = ds.dataset([str(path) for path in files], format="parquet")
    keys = set()
    for batch in dataset.scanner(columns=["source_id", "record_id", "content_hash"]).to_batches():
        for row in batch.to_pylist():
            keys.add(_key(row))
    return keys


def _next_part_index(output_dir: Path, shard_id: str) -> int:
    existing = sorted(output_dir.glob(f"part-shard-{shard_id}-*.parquet"))
    return len(existing)


def _key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("source_id") or ""),
        str(row.get("record_id") or ""),
        str(row.get("content_hash") or ""),
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())

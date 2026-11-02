"""Prepare all English source rows for downstream classification, without selection.

Streaming per-shard Parquet outputs retain text and source metadata. A separate
exact-content index supplies one read location per text for inference; it does
not choose the attribution or source precedence of a released document.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import logging
import multiprocessing
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken

from ingest import utils
from . import download, profile

LOG = logging.getLogger("youtube-english")
VERSION = "youtube-english-candidates-v1"
SCHEMA = pa.schema([
    ("dataset_revision", pa.string()), ("source_shard", pa.string()),
    ("source_row", pa.int64()), ("text", pa.large_string()),
    ("content_hash", pa.string()), ("content_length", pa.int64()),
    ("text_state", pa.string()), ("metadata_json", pa.large_string()),
    *[(name, pa.string()) for name in profile.TEXT_COLUMNS],
])


def _string(value):
    return value if isinstance(value, str) or value is None else json.dumps(value, default=str, sort_keys=True)


def _row(row, shard, index, revision):
    text = row.get("text")
    if text is not None and not isinstance(text, str):
        raise ValueError(f"Non-string transcript: {shard}:{index}")
    state = "missing" if text is None else "blank" if not text.strip() else "candidate"
    return {"dataset_revision": revision, "source_shard": shard, "source_row": index,
        "text": text, "content_hash": utils.compute_content_hash(text) if state == "candidate" else None,
        "content_length": utils.compute_token_count(text) if state == "candidate" else 0,
        "text_state": state,
        "metadata_json": json.dumps({k:v for k,v in row.items() if k != "text"}, ensure_ascii=False, default=str),
        **{name:_string(row.get(name)) for name in profile.TEXT_COLUMNS}}


def prepare_shard(task):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    index, entry, root, output, revision, config_sha = task
    pa.set_cpu_count(1)
    source = download._shard_path(root / "raw", entry)
    before = profile._source_stat(source)
    raw = download._verify_file(source, entry)
    target = output / "rows" / f"{index:05d}.parquet"
    checkpoint = output / "checkpoints" / f"{index:05d}.json"
    binding = {"source_shard":entry["path"], "raw_sha256":raw["sha256"], "run_config_sha256":config_sha}
    if checkpoint.exists():
        try:
            saved = json.loads(checkpoint.read_text())
            if (all(saved.get(k) == v for k,v in binding.items())
                    and profile._sha256(target) == saved["output_sha256"]):
                return {**saved, "resumed":True}
        except (OSError, ValueError, KeyError):
            pass
    temporary = target.with_suffix(".tmp")
    counts = {"raw_rows":0, "english_rows":0, "candidate_rows":0,
              "blank_rows":0, "missing_rows":0, "candidate_tokens":0}
    try:
        with pq.ParquetFile(source) as parquet, pq.ParquetWriter(temporary, SCHEMA, compression="zstd") as writer:
            fields = {f.name:str(f.type) for f in parquet.schema_arrow}
            if fields.get("text") not in {"string", "large_string", "null"}:
                raise ValueError(f"Unexpected text schema: {entry['path']}")
            if fields.get("transcription_language", "null") not in {"string", "large_string", "null"}:
                raise ValueError(f"Unexpected language schema: {entry['path']}")
            # Small batches bound text memory even for exceptionally long videos.
            selected, buffered_chars = [], 0
            for batch in parquet.iter_batches(batch_size=16, use_threads=False):
                for row in batch.to_pylist():
                    row_index = counts["raw_rows"]
                    counts["raw_rows"] += 1
                    # Approved frame: English, including translated English rows.
                    if row.get("transcription_language") != "en":
                        continue
                    item = _row(row, entry["path"], row_index, revision)
                    selected.append(item)
                    buffered_chars += len(item['text'] or '')
                    counts["english_rows"] += 1
                    counts[item["text_state"] + "_rows"] += 1
                    counts["candidate_tokens"] += item["content_length"]
                # I/O buffer limits only; never document eligibility limits.
                if len(selected) >= 512 or buffered_chars >= 2 * 1024 * 1024:
                    writer.write_table(pa.Table.from_pylist(selected, schema=SCHEMA))
                    selected, buffered_chars = [], 0
                if counts["raw_rows"] % 10000 < 16:
                    LOG.info("%s: %s source rows; %s English rows", entry["path"], counts["raw_rows"], counts["english_rows"])
            if selected:
                writer.write_table(pa.Table.from_pylist(selected, schema=SCHEMA))
            if counts["raw_rows"] != parquet.metadata.num_rows:
                raise ValueError("Incomplete source scan")
        if before != profile._source_stat(source):
            raise ValueError(f"Source changed while preparing: {entry['path']}")
        temporary.replace(target)
        result = {**binding, **counts, "output_sha256":profile._sha256(target)}
        download._write_json(checkpoint, result)
        return result
    finally:
        temporary.unlink(missing_ok=True)


def summarize(output, results, workers, expected_english):
    files = [str(output / "rows" / f"{i:05d}.parquet") for i in range(len(results))]
    target = output / "unique_index.parquet"
    temporary = target.with_suffix(".tmp")
    try:
        with duckdb.connect(config={"memory_limit":"4GB", "threads":workers,
                                   "temp_directory":str(output / "duckdb-temp")}) as con:
            con.read_parquet(files, hive_partitioning=False).create_view("rows")
            counts = con.execute("""SELECT count(*), count(*) FILTER (WHERE text_state='candidate'),
                coalesce(sum(content_length),0) FROM rows""").fetchone()
            if counts != (sum(r['english_rows'] for r in results), sum(r['candidate_rows'] for r in results),
                          sum(r['candidate_tokens'] for r in results)):
                raise ValueError("Shard totals do not reproduce from outputs")
            if counts[0] != expected_english:
                raise ValueError(f"English coverage differs from completed profile: {counts[0]} != {expected_english}")
            inconsistent = con.execute("""SELECT count(*) FROM (
                SELECT content_hash FROM rows WHERE text_state='candidate'
                GROUP BY content_hash HAVING min(content_length) <> max(content_length))""").fetchone()[0]
            if inconsistent:
                raise ValueError("Conflicting token lengths for an identical content hash")
            con.execute("""CREATE TEMP VIEW exact_texts AS SELECT content_hash,
                max(content_length) content_length, count(*) source_copies,
                arg_min(struct_pack(source_shard := source_shard, source_row := source_row),
                        struct_pack(source_shard := source_shard, source_row := source_row)) read_location
                FROM rows WHERE text_state='candidate' GROUP BY content_hash""")
            unique_count, unique_tokens = con.execute("SELECT count(*), coalesce(sum(content_length),0) FROM exact_texts").fetchone()
            con.execute(f"COPY (SELECT content_hash, content_length, source_copies, read_location.source_shard source_shard, "
                        f"read_location.source_row source_row FROM exact_texts ORDER BY content_hash) TO "
                        f"{profile._sql(str(temporary))} (FORMAT PARQUET, COMPRESSION ZSTD)")
            languages = profile._records(con.execute("""SELECT original_language, count(*) records,
                sum(content_length) candidate_tokens FROM rows GROUP BY original_language ORDER BY original_language"""))
        temporary.replace(target)
        return {"english_rows":counts[0], "candidate_rows":counts[1], "candidate_tokens":counts[2],
            "exact_unique_texts":unique_count, "exact_unique_tokens":unique_tokens,
            "extra_exact_copies":counts[1]-unique_count, "original_languages":languages,
            "unique_index_sha256":profile._sha256(target)}
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be positive")
    root = args.data_dir.resolve()
    output = (args.output_dir or root / "english-candidates-v1").resolve()
    forbidden = [root / "raw", root / "profile-v1"]
    if output == root or root.is_relative_to(output) or any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in forbidden):
        parser.error("Output must be separate from raw data and the completed profile")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with download._directory_lock(root):
        manifest, manifest_sha = profile._load_inputs(root)
        profile_path = root / "profile-v1" / "summary.json"
        inventory = json.loads(profile_path.read_text())
        profile_config = json.loads((profile_path.parent / "run-config.json").read_text())
        if (inventory.get("complete") is not True or inventory["dataset_revision"] != manifest["revision"]
                or profile_config["manifest_sha256"] != manifest_sha):
            raise ValueError("A matching completed profile is required")
        expected_english = sum(r['records'] for r in inventory['transcription_languages'] if r['transcription_language']=='en')
        config = {"version":VERSION, "manifest_sha256":manifest_sha, "profile_summary_sha256":profile._sha256(profile_path),
            "dataset_revision":manifest["revision"], "language":"en", "quality_filters":None,
            "code":{Path(m.__file__).name:profile._sha256(Path(m.__file__)) for m in (download, profile, utils)},
            "runner_sha256":profile._sha256(Path(__file__)), "tokenizer_sha256":profile._sha256(utils._TOKENIZER_ASSET),
            "versions":{"pyarrow":pa.__version__, "duckdb":duckdb.__version__, "tiktoken":tiktoken.__version__}}
        output.mkdir(parents=True, exist_ok=True)
        with download._directory_lock(output):
            config_path = output / "run-config.json"
            if config_path.exists() and json.loads(config_path.read_text()) != config:
                raise ValueError("Preparation configuration changed; choose a new output directory")
            download._write_json(config_path, config)
            config_sha = profile._sha256(config_path)
            for name in ('rows','checkpoints'):
                (output / name).mkdir(exist_ok=True)
            summary_path = output / "summary.json"
            download._write_json(summary_path, {"complete":False, "status":"preparing", "run_config_sha256":config_sha})
            tasks = [(i,e,root,output,manifest['revision'],config_sha) for i,e in enumerate(manifest['files'])]
            results = [None] * len(tasks)
            LOG.info("Preparing all %s English rows from %d shards; no quality selection", f"{expected_english:,}", len(tasks))
            if args.workers == 1:
                for i,task in enumerate(tasks):
                    results[i] = prepare_shard(task)
                    LOG.info("[%d/%d] %s", i+1, len(tasks), results[i]['source_shard'])
            else:
                with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                    futures = {pool.submit(prepare_shard,t):i for i,t in enumerate(tasks)}
                    for done,future in enumerate(as_completed(futures),1):
                        i = futures[future]
                        results[i] = future.result()
                        LOG.info("[%d/%d] %s%s", done, len(tasks), results[i]['source_shard'], " (resumed)" if results[i].get('resumed') else '')
            totals = summarize(output, results, args.workers, expected_english)
            report = {"complete":True, "version":VERSION, "dataset_revision":manifest['revision'],
                "run_config_sha256":config_sha, "completed_at":datetime.now(timezone.utc).isoformat(),
                "tokenizer":"cl100k_base", **totals, "shards":[{k:v for k,v in r.items() if k != 'resumed'} for r in results],
                "scope":"All metadata-labeled English rows including translations. Text unchanged; no semantic selection. "
                        "Blank/missing rows retained for accounting. Exact index is an inference read pointer, not release precedence. "
                        "Candidate totals are not retained corpus tokens."}
            download._write_json(summary_path, report)
            LOG.info("ENGLISH READY: %s distinct texts, %s exact-unique candidate tokens; %s", f"{totals['exact_unique_texts']:,}", f"{totals['exact_unique_tokens']:,}", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

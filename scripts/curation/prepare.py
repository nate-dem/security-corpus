"""Materialize exact-unique inference inputs from completed source inventories.

No semantic filtering, release precedence, or canonical schema change. Keep every
valid source alias in lineage.parquet. Texts are deduplicated within prompt kind
(web/youtube); cross-kind duplicates remain visible for final corpus deduplication.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import json
import logging
import multiprocessing
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from ingest.utils import compute_content_hash
from scripts.youtube import download, profile as yt
from scripts.web_corpora import compare, profile as web
from scripts.web_corpora.sources import data_files

VERSION = "curation-inputs-v1"
# I/O partition sizes, not content eligibility/length thresholds. Oversized records stay whole.
PART_RECORDS = 2048
PART_CHARACTERS = 16 * 1024 * 1024
SCHEMA = pa.schema(
    [
        ("source", pa.string()),
        ("kind", pa.string()),
        ("dataset_revision", pa.string()),
        ("source_shard", pa.string()),
        ("source_row", pa.int64()),
        ("content_hash", pa.string()),
        ("content_length", pa.int64()),
        ("source_copies", pa.int64()),
        ("text", pa.large_string()),
    ]
)
LOG = logging.getLogger("curation-prepare")


def inventory(youtube, web_root):
    inputs = []
    report = json.loads((youtube / "summary.json").read_text())
    if (
        report.get("complete") is not True
        or report["run_config_sha256"] != yt._sha256(youtube / "run-config.json")
        or report["unique_index_sha256"] != yt._sha256(youtube / "unique_index.parquet")
    ):
        raise ValueError("English preparation must be complete and unchanged")
    for i, entry in enumerate(report["shards"]):
        path = youtube / "rows" / f"{i:05d}.parquet"
        if yt._sha256(path) != entry["output_sha256"]:
            raise ValueError(f"English input checksum mismatch: {path}")
        inputs.append(
            {
                "source": "youtube-commons",
                "kind": "youtube",
                "index": i,
                "dataset_revision": report["dataset_revision"],
                "source_shard": entry["source_shard"],
                "path": str(path),
                "sha256": entry["output_sha256"],
                "metadata": str(path),
                "rows": entry["english_rows"],
                "valid_rows": entry["candidate_rows"],
                "tokens": entry["candidate_tokens"],
            }
        )
    bindings = {"youtube-commons": yt._sha256(youtube / "summary.json")}
    for source, name in (
        ("redsage-cfw", "profile-v1"),
        ("primus-fineweb", "profile-v2"),
    ):
        root = web_root / source
        paths = compare.profile_paths(root, name, source)
        manifest, _ = web.load_inputs(root, source)
        report = json.loads((root / name / "summary.json").read_text())
        bindings[source] = yt._sha256(root / name / "summary.json")
        for i, (entry, metadata, shard) in enumerate(
            zip(data_files(manifest), paths, report["shards"])
        ):
            path = download._shard_path(root / "raw", entry)
            inputs.append(
                {
                    "source": source,
                    "kind": "web",
                    "index": i,
                    "dataset_revision": manifest["revision"],
                    "source_shard": entry["path"],
                    "path": str(path),
                    "sha256": shard["raw_sha256"],
                    "metadata": metadata,
                    "rows": shard["rows"],
                }
            )
    return inputs, bindings


def build_index(inputs, output, threads):
    """Aggregate narrow metadata using DuckDB spill; never sort full source text."""
    views = []
    with duckdb.connect(
        config={
            "memory_limit": "8GB",
            "threads": threads,
            "temp_directory": str(output / "duckdb-temp"),
        }
    ) as con:
        for source in sorted({i["source"] for i in inputs}):
            group = [i for i in inputs if i["source"] == source]
            view = "input_" + str(len(views))
            con.read_parquet(
                [i["metadata"] for i in group], hive_partitioning=False
            ).create_view(view)
            locations = pa.Table.from_pylist(
                [
                    {
                        "source_shard": i["source_shard"],
                        "shard_index": i["index"],
                        "dataset_revision": i["dataset_revision"],
                    }
                    for i in group
                ]
            )
            con.register("locations", locations)
            kind = group[0]["kind"]
            state = "candidate" if kind == "youtube" else "valid"
            # Snapshot small locator table before reusing its registration.
            con.execute(f"CREATE TEMP TABLE loc_{view} AS SELECT * FROM locations")
            if con.execute(
                f"SELECT count(*) FROM {view} v ANTI JOIN loc_{view} l USING(source_shard)"
            ).fetchone()[0]:
                raise ValueError("Metadata contains an unknown shard")
            views.append(
                f"SELECT {yt._sql(source)} AS source, {yt._sql(kind)} AS kind, l.shard_index, l.dataset_revision, "
                f"v.source_shard, v.source_row, v.content_hash, v.content_length "
                f"FROM {view} v JOIN loc_{view} l USING(source_shard) WHERE v.text_state={yt._sql(state)}"
            )
        con.execute("CREATE TEMP VIEW aliases AS " + " UNION ALL ".join(views))
        bad = con.execute("""SELECT count(*) FROM aliases WHERE source_row IS NULL OR source_row<0 OR
            content_hash IS NULL OR NOT regexp_full_match(content_hash,'[0-9a-f]{64}') OR
            content_length IS NULL OR content_length<=0""").fetchone()[0]
        if bad:
            raise ValueError("Invalid candidate identities or token counts")
        if con.execute(
            "SELECT count(*) FROM (SELECT source,source_shard,source_row FROM aliases GROUP BY ALL HAVING count(*)<>1)"
        ).fetchone()[0]:
            raise ValueError("Duplicate source row identity in inventory")
        if con.execute(
            "SELECT count(*) FROM (SELECT content_hash FROM aliases GROUP BY content_hash HAVING min(content_length)<>max(content_length))"
        ).fetchone()[0]:
            raise ValueError("Identical text hashes have conflicting token counts")
        con.execute("""CREATE TEMP TABLE unique_inputs AS SELECT kind,content_hash,max(content_length) content_length,
            count(*) source_copies,arg_min(struct_pack(source:=source,shard_index:=shard_index,
            source_shard:=source_shard,source_row:=source_row,dataset_revision:=dataset_revision),
            struct_pack(source:=source,shard_index:=shard_index,source_row:=source_row)) loc
            FROM aliases GROUP BY kind,content_hash""")
        for name, query in [
            ("lineage", "SELECT * FROM aliases ORDER BY source,shard_index,source_row"),
            (
                "read_locations",
                "SELECT kind,content_hash,content_length,source_copies,loc.* FROM unique_inputs ORDER BY loc.source,loc.shard_index,loc.source_row",
            ),
        ]:
            tmp = output / f"{name}.tmp"
            con.execute(
                f"COPY ({query}) TO {yt._sql(str(tmp))} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 16384)"
            )
            tmp.replace(output / f"{name}.parquet")
        totals = web._query_rows(
            con,
            "SELECT kind,count(*) texts,sum(content_length) tokens,sum(source_copies) aliases FROM unique_inputs GROUP BY kind ORDER BY kind",
        )
        global_total = web._query_rows(
            con,
            "SELECT count(*) texts,sum(tokens) tokens FROM (SELECT content_hash,max(content_length) tokens FROM unique_inputs GROUP BY content_hash)",
        )[0]
        return {
            "by_kind": totals,
            "across_kind_exact_unique": global_total,
            "lineage_sha256": yt._sha256(output / "lineage.parquet"),
            "read_locations_sha256": yt._sha256(output / "read_locations.parquet"),
        }


def materialize(task):
    item, output, config_sha = task
    pa.set_cpu_count(1)
    path = Path(item["path"])
    before = web._verify_stat(path)
    if yt._sha256(path) != item["sha256"]:
        raise ValueError(f"Source checksum changed: {path}")
    folder = output / "shards" / item["source"] / f"{item['index']:05d}"
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / "complete.json"
    if checkpoint.exists():
        cached = json.loads(checkpoint.read_text())
        if (
            cached["run_config_sha256"] == config_sha
            and cached["input_sha256"] == item["sha256"]
            and all(
                yt._sha256(output / p["path"]) == p["sha256"] for p in cached["parts"]
            )
        ):
            return cached
    pointers = (
        ds.dataset(output / "read_locations.parquet")
        .to_table(
            filter=(ds.field("source") == item["source"])
            & (ds.field("shard_index") == item["index"])
        )
        .to_pylist()
    )
    wanted = {r["source_row"]: r for r in pointers}
    expected = len(wanted)
    writer = None
    buffer = []
    chars = 0
    buffered_chars = 0
    records = 0
    tokens = 0
    parts = []
    seen_rows = 0
    tmp = None

    def flush():
        nonlocal buffer, buffered_chars
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=SCHEMA))
            buffer = []
            buffered_chars = 0

    def close():
        nonlocal writer, records, tokens, chars
        if writer is not None:
            flush()
            writer.close()
            writer = None
            target = folder / f"{len(parts):05d}.parquet"
            tmp.replace(target)
            parts.append(
                {
                    "path": str(target.relative_to(output)),
                    "sha256": yt._sha256(target),
                    "records": records,
                    "tokens": tokens,
                    "characters": chars,
                }
            )
            records = tokens = chars = 0

    try:
        # Even empty pointer sets validate the source bytes; skip decompression.
        rows = web.iter_rows(path) if wanted else ()
        for index, row in enumerate(rows):
            seen_rows += 1
            source_row = row["source_row"] if item["kind"] == "youtube" else index
            pointer = wanted.pop(source_row, None)
            if pointer is None:
                continue
            text = row.get(
                "text" if item["kind"] == "youtube" else web.text_field(item["source"])
            )
            if (
                not isinstance(text, str)
                or not text.strip()
                or compute_content_hash(text) != pointer["content_hash"]
            ):
                raise ValueError(f"Text/read pointer mismatch: {path}:{source_row}")
            if writer is None:
                tmp = folder / f"{len(parts):05d}.tmp"
                writer = pq.ParquetWriter(tmp, SCHEMA, compression="zstd")
            buffer.append(
                {k: pointer[k] for k in SCHEMA.names if k != "text"} | {"text": text}
            )
            records += 1
            tokens += pointer["content_length"]
            chars += len(text)
            buffered_chars += len(text)
            if len(buffer) >= 64 or buffered_chars >= 2 * 1024 * 1024:
                flush()
            if records >= PART_RECORDS or chars >= PART_CHARACTERS:
                close()
        if expected and seen_rows != item["rows"]:
            raise ValueError(f"Incomplete source scan: {path}")
        if wanted or before != web._verify_stat(path):
            raise ValueError("Missing read pointers or input changed during scan")
        close()
        result = {
            "run_config_sha256": config_sha,
            "input_sha256": item["sha256"],
            "source": item["source"],
            "index": item["index"],
            "records": sum(p["records"] for p in parts),
            "tokens": sum(p["tokens"] for p in parts),
            "parts": parts,
        }
        if result["records"] != expected:
            raise ValueError("Output coverage mismatch")
        download._write_json(checkpoint, result)
        return result
    finally:
        if writer is not None:
            writer.close()
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--youtube", type=Path, required=True)
    parser.add_argument("--web-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    youtube = args.youtube.resolve()
    web_root = args.web_root.resolve()
    if args.workers < 1 or any(
        output == p or output.is_relative_to(p) or p.is_relative_to(output)
        for p in (youtube, web_root)
    ):
        parser.error("Use positive workers and an output separate from source trees")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for p in (
            output,
            youtube,
            web_root / "redsage-cfw",
            web_root / "primus-fineweb",
        ):
            stack.enter_context(download._directory_lock(p))
        inputs, bindings = inventory(youtube, web_root)
        config = {
            "version": VERSION,
            "source_summaries": bindings,
            "inputs": inputs,
            "runner_sha256": yt._sha256(Path(__file__)),
            "part_records": PART_RECORDS,
            "part_characters": PART_CHARACTERS,
            "code": {
                m.__name__: yt._sha256(Path(m.__file__))
                for m in (web, compare, download, yt)
            },
            "versions": {"pyarrow": pa.__version__, "duckdb": duckdb.__version__},
        }
        config_path = output / "run-config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError(
                "Inputs or implementation changed; use a new output directory"
            )
        download._write_json(config_path, config)
        config_sha = yt._sha256(config_path)
        download._write_json(
            output / "summary.json", {"complete": False, "stage": "indexing"}
        )
        index_path = output / "index-summary.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())
            if (
                index["run_config_sha256"] != config_sha
                or index["lineage_sha256"] != yt._sha256(output / "lineage.parquet")
                or index["read_locations_sha256"]
                != yt._sha256(output / "read_locations.parquet")
            ):
                raise ValueError("Index checksum mismatch")
        else:
            index = {
                **build_index(inputs, output, args.workers),
                "run_config_sha256": config_sha,
            }
            download._write_json(index_path, index)
        LOG.info("Index ready: %s; materializing texts", index["by_kind"])
        tasks = [(i, output, config_sha) for i in inputs]
        results = []
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [pool.submit(materialize, t) for t in tasks]
            for n, f in enumerate(as_completed(futures), 1):
                r = f.result()
                results.append(r)
                LOG.info(
                    "[%d/%d] %s/%05d: %d unique texts",
                    n,
                    len(tasks),
                    r["source"],
                    r["index"],
                    r["records"],
                )
        results.sort(key=lambda r: (r["source"], r["index"]))
        totals = {k: sum(r[k] for r in results) for k in ("records", "tokens")}
        if totals != {
            "records": sum(r["texts"] for r in index["by_kind"]),
            "tokens": sum(r["tokens"] for r in index["by_kind"]),
        }:
            raise ValueError("Materialization totals do not match index")
        summary = {
            "complete": True,
            "version": VERSION,
            "run_config_sha256": config_sha,
            **totals,
            "index": index,
            "parts": [p for r in results for p in r["parts"]],
            "scope": "Inference inputs only. No semantic selection or release precedence. All aliases remain in lineage.parquet. "
            "Cross-kind and baseline overlap not removed; tokens are not retained additions.",
        }
        download._write_json(output / "summary.json", summary)
        LOG.info(
            "PREPARATION COMPLETE: %d inference texts, %d tokens, %d parts",
            totals["records"],
            totals["tokens"],
            len(summary["parts"]),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

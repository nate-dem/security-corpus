"""Resumable CPU near-deduplication of approved web additions against baseline.

Source Parquet is read-only. SQLite indexes only retained representatives, so
every removal has a direct verified witness. Transactions checkpoint decisions;
bounded signature tasks run in parallel before sequential representative choice.
"""

import argparse
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import importlib.metadata
from itertools import zip_longest
import json
import logging
import multiprocessing
from pathlib import Path
import random
import sqlite3
import sys
import time
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ingest.utils import compute_content_hash, compute_token_count
from scripts.near_dedup import core
from scripts.youtube.download import _directory_lock, _write_json
from scripts.youtube.profile import _sha256

HERE = Path(__file__).resolve().parent
BATCH = 128
STRIDE = 1 << 32
LOG = logging.getLogger(__name__)
SKETCH_SCHEMA = pa.schema(
    [
        ("doc_id", pa.int64()),
        ("source", pa.string()),
        ("tokens", pa.int64()),
        ("content_hash", pa.string()),
        ("shingle_count", pa.int64()),
        ("signature", pa.binary(1024)),
        ("text_zlib", pa.binary()),
        ("origin", pa.string()),
    ]
)
DECISION_SCHEMA = pa.schema(
    [
        ("doc_id", pa.int64()),
        ("source", pa.string()),
        ("content_hash", pa.string()),
        ("tokens", pa.int64()),
        ("keep", pa.bool_()),
        ("reason", pa.string()),
        ("representative_id", pa.int64()),
        ("representative_source", pa.string()),
        ("representative_hash", pa.string()),
        ("intersection", pa.int64()),
        ("union", pa.int64()),
        ("jaccard", pa.float64()),
        ("origin", pa.string()),
        ("representative_origin", pa.string()),
        ("plan_sha256", pa.string()),
    ]
)


def read(path):
    return json.loads(Path(path).read_text())


def runtime():
    version = importlib.metadata.version("datasketch")
    if version != "1.6.5":
        raise ValueError("Install the pinned datasketch==1.6.5")
    return {
        "python": sys.version.split()[0],
        "datasketch": version,
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
        "sqlite": sqlite3.sqlite_version,
        "permutations_sha256": core.permutation_digest(),
    }


def checked(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Input path leaves its bound directory")
    return path


def code_hashes():
    paths = list(HERE.glob("*.py")) + list(HERE.glob("*.json"))
    paths += [Path(sys.modules[compute_content_hash.__module__].__file__)]
    return {str(p.relative_to(HERE.parent.parent)): _sha256(p) for p in sorted(paths)}


def make_plan(project, selected, output, references=None):
    project, selected, output = (Path(p).resolve() for p in (project, selected, output))
    for protected in (project, selected):
        if (
            output == protected
            or output.is_relative_to(protected)
            or protected.is_relative_to(output)
        ):
            raise ValueError(
                "Keep dedup output separate from project and selected inputs"
            )
    refs = references if references is not None else read(HERE / "inputs.json")
    if (
        _sha256(selected / "summary.json") != refs["selected_summary_sha256"]
        or _sha256(selected / "selection.json") != refs["selected_config_sha256"]
    ):
        raise ValueError("Selected input differs from reviewed evidence")
    summary, binding = (
        read(selected / "summary.json"),
        read(selected / "selection.json"),
    )
    if (
        not summary["complete"]
        or summary != read(selected / "review/summary.json")
        or binding["min_domain_and_substance_score"] != 0.80
        or summary["plan_sha256"] != binding["plan_sha256"]
    ):
        raise ValueError("Incomplete or incompatible selected inputs")
    files = []
    for f in sorted(refs["baseline_files"], key=lambda f: f["project_relative_path"]):
        files.append(
            {
                "path": str(checked(project, f["project_relative_path"])),
                "sha256": f["sha256"],
                "records": f["records"],
                "tokens": f["recomputed_tokens"],
                "baseline": True,
            }
        )
    baseline_count = len(files)
    for f in sorted(summary["files"], key=lambda f: f["index"]):
        files.append(
            {
                "path": str(checked(selected, f["path"])),
                "sha256": f["sha256"],
                "metadata_path": str(checked(selected, f["metadata_path"])),
                "metadata_sha256": f["metadata_sha256"],
                "source": f["source"],
                "records": f["records"],
                "tokens": f["tokens"],
                "baseline": False,
            }
        )
    if len({f["path"] for f in files}) != len(files):
        raise ValueError("Duplicate input file")
    if any(not 0 < f["records"] < STRIDE or f["tokens"] <= 0 for f in files):
        raise ValueError("Invalid input size")
    if (
        sum(f["records"] for f in files[:baseline_count]) != refs["baseline_records"]
        or sum(f["tokens"] for f in files[:baseline_count]) != refs["baseline_tokens"]
    ):
        raise ValueError("Baseline inventory counts disagree")
    for source, counts in summary["by_source"].items():
        for field in ("records", "tokens"):
            if (
                sum(f[field] for f in files if f.get("source") == source)
                != counts[field]
            ):
                raise ValueError("Web inventory counts disagree")
    plan = {
        "policy": core.POLICY,
        "runtime": runtime(),
        "code": code_hashes(),
        "references": refs,
        "project": str(project),
        "selected": str(selected),
        "output": str(output),
        "files": files,
        "signature_batch_size": BATCH,
        "selected_by_source": summary["by_source"],
    }
    path = output / "plan.json"
    if path.exists() and read(path) != plan:
        raise ValueError("Frozen dedup plan changed; restore inputs/runtime/code")
    _write_json(path, plan)
    return plan


def verify_inputs(plan):
    for i, file in enumerate(plan["files"]):
        for path_key, sha_key in (
            ("path", "sha256"),
            ("metadata_path", "metadata_sha256"),
        ):
            if path_key in file and _sha256(Path(file[path_key])) != file[sha_key]:
                raise ValueError(f"Input checksum mismatch: {file[path_key]}")
        if i % 1000 == 0:
            LOG.info("Verified %s/%s input files", i, len(plan["files"]))


def write_parquet(path, rows, schema):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd"
    )
    temporary.replace(path)


def location(root, file_index, start):
    return root / "signatures" / f"{file_index:05d}" / f"{start:010d}.parquet"


def sketch_task(task):
    path, digest, rows = task
    result = []
    for row in rows:
        text = row.pop("text")
        if (
            compute_content_hash(text) != row["content_hash"]
            or compute_token_count(text) != row["tokens"]
        ):
            raise ValueError(f"Text hash/token mismatch: {row['doc_id']}")
        values = core.shingles(text)
        result.append(
            {
                **row,
                "shingle_count": len(values),
                "signature": core.signature(values),
                "text_zlib": zlib.compress(text.encode("utf-8"), level=1),
            }
        )
    write_parquet(path, result, SKETCH_SCHEMA)
    report = {
        "plan_sha256": digest,
        "sha256": _sha256(path),
        "records": len(result),
        "tokens": sum(r["tokens"] for r in result),
    }
    _write_json(path.with_suffix(".json"), report)
    return report


def cached_signature(path, digest, expected_records):
    marker = path.with_suffix(".json")
    if not marker.exists():
        return None
    report = read(marker)
    if (
        report["plan_sha256"] != digest
        or report["records"] != expected_records
        or _sha256(path) != report["sha256"]
    ):
        raise ValueError("Signature checkpoint changed")
    return report


def input_rows(file, file_index):
    path = Path(file["path"])
    record_stream = pq.ParquetFile(path).iter_batches(
        batch_size=BATCH, use_threads=False
    )
    metadata_stream = (
        None
        if file["baseline"]
        else pq.ParquetFile(file["metadata_path"]).iter_batches(
            batch_size=BATCH, use_threads=False
        )
    )
    pairs = (
        ((batch, None) for batch in record_stream)
        if metadata_stream is None
        else zip_longest(record_stream, metadata_stream)
    )
    start = 0
    for batch, meta in pairs:
        if batch is None or (not file["baseline"] and meta is None):
            raise ValueError("Text/metadata coverage mismatch")
        rows, metadata = (
            batch.to_pylist(),
            meta.to_pylist() if meta is not None else None,
        )
        if metadata is not None and len(rows) != len(metadata):
            raise ValueError("Text/metadata batch mismatch")
        result = []
        for j, row in enumerate(rows):
            source = row["source_id"] if file["baseline"] else row["source"]
            origin = {
                k: row.get(k)
                for k in (
                    "record_id",
                    "source_record_id",
                    "source_url",
                    "license",
                    "dataset_revision",
                    "source_shard",
                    "source_row",
                )
            }
            if metadata is not None:
                m = metadata[j]
                if source != file["source"] or any(
                    row[k] != m[k]
                    for k in (
                        "source",
                        "content_hash",
                        "content_length",
                        "source_shard",
                        "source_row",
                        "dataset_revision",
                    )
                ):
                    raise ValueError("Text/metadata identity mismatch")
                origin.update(
                    {
                        k: m.get(k)
                        for k in ("url", "upstream_license", "upstream_relevant")
                    }
                )
            origin.update(
                input_path=file["path"], row=start + j, input_sha256=file["sha256"]
            )
            result.append(
                {
                    "doc_id": file_index * STRIDE + start + j,
                    "source": source,
                    "tokens": row["content_length"],
                    "content_hash": row["content_hash"],
                    "text": row["content"] if file["baseline"] else row["text"],
                    "origin": json.dumps(origin, sort_keys=True),
                }
            )
        yield start, result
        start += len(rows)
    if start != file["records"]:
        raise ValueError("Source record count mismatch")


def signature_stage(plan, digest, workers):
    root = Path(plan["output"])
    pending = set()
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for i, file in enumerate(plan["files"]):
            missing, totals = set(), Counter()
            for start in range(0, file["records"], BATCH):
                path = location(root, i, start)
                report = cached_signature(
                    path, digest, min(BATCH, file["records"] - start)
                )
                if report is None:
                    missing.add(start)
                else:
                    totals.update({k: report[k] for k in ("records", "tokens")})
            if not missing:
                if totals != Counter({k: file[k] for k in ("records", "tokens")}):
                    raise ValueError("Signature/source totals mismatch")
                continue
            for start, rows in input_rows(file, i):
                if start not in missing:
                    continue
                pending.add(
                    pool.submit(sketch_task, (location(root, i, start), digest, rows))
                )
                if len(pending) >= workers * 2:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        future.result()
            if i % 100 == 0:
                LOG.info("Signature input file %s/%s", i, len(plan["files"]))
                _write_json(
                    root / "progress.json",
                    {
                        "stage": "signatures",
                        "input_file": i,
                        "total_files": len(plan["files"]),
                        "complete": False,
                    },
                )
        for future in pending:
            future.result()


def database(root, digest):
    con = sqlite3.connect(root / "state.sqlite")
    con.execute("PRAGMA journal_mode=DELETE")
    con.execute("PRAGMA synchronous=FULL")
    con.execute("PRAGMA cache_size=-524288")
    con.execute("PRAGMA temp_store=FILE")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS binding (plan TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS loaded (path TEXT PRIMARY KEY, sha TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS docs (
            doc_id INTEGER PRIMARY KEY, baseline INTEGER NOT NULL, source TEXT NOT NULL,
            tokens INTEGER NOT NULL, content_hash TEXT NOT NULL, shingle_count INTEGER NOT NULL,
            signature BLOB NOT NULL, text_zlib BLOB NOT NULL, origin TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS decisions (
            doc_id INTEGER PRIMARY KEY, keep INTEGER NOT NULL, reason TEXT NOT NULL,
            rep INTEGER NOT NULL, intersection INTEGER NOT NULL, union_size INTEGER NOT NULL,
            comparisons INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS exact (hash TEXT PRIMARY KEY, doc_id INTEGER NOT NULL) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS bands (
            band INTEGER NOT NULL, bucket BLOB NOT NULL, doc_id INTEGER NOT NULL,
            PRIMARY KEY (band, bucket, doc_id)) WITHOUT ROWID;
    """)
    binding = con.execute("SELECT plan FROM binding").fetchall()
    if binding and binding != [(digest,)]:
        raise ValueError("Database plan binding differs")
    if not binding:
        con.execute("INSERT INTO binding VALUES (?)", (digest,))
    con.commit()
    if con.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
        raise ValueError("SQLite integrity check failed")
    return con


def load_signatures(con, plan, digest):
    root = Path(plan["output"])
    for i, file in enumerate(plan["files"]):
        totals = Counter()
        for start in range(0, file["records"], BATCH):
            path = location(root, i, start)
            report = cached_signature(path, digest, min(BATCH, file["records"] - start))
            if report is None:
                raise ValueError("Missing signature partition")
            totals.update({k: report[k] for k in ("records", "tokens")})
            relative = str(path.relative_to(root))
            loaded = con.execute(
                "SELECT sha FROM loaded WHERE path=?", (relative,)
            ).fetchone()
            if loaded:
                if loaded[0] != report["sha256"]:
                    raise ValueError("Loaded signature checksum changed")
                continue
            rows = pq.ParquetFile(path).read().to_pylist()
            if [r["doc_id"] for r in rows] != list(
                range(i * STRIDE + start, i * STRIDE + start + len(rows))
            ):
                raise ValueError("Signature row identity changed")
            with con:
                con.executemany(
                    "INSERT INTO docs VALUES (?,?,?,?,?,?,?,?,?)",
                    [
                        (
                            r["doc_id"],
                            int(file["baseline"]),
                            r["source"],
                            r["tokens"],
                            r["content_hash"],
                            r["shingle_count"],
                            r["signature"],
                            r["text_zlib"],
                            r["origin"],
                        )
                        for r in rows
                    ],
                )
                con.execute(
                    "INSERT INTO loaded VALUES (?,?)", (relative, report["sha256"])
                )
        if totals != Counter({k: file[k] for k in ("records", "tokens")}):
            raise ValueError("Signature/source totals mismatch")
        if i % 1000 == 0:
            LOG.info("Loaded %s/%s files into disk index", i, len(plan["files"]))
            _write_json(
                root / "progress.json",
                {
                    "stage": "load_index",
                    "input_file": i,
                    "total_files": len(plan["files"]),
                    "complete": False,
                },
            )
    expected = (
        sum(f["records"] for f in plan["files"]),
        sum(f["tokens"] for f in plan["files"]),
    )
    if con.execute("SELECT count(*),sum(tokens) FROM docs").fetchone() != expected:
        raise ValueError("Disk index/input totals mismatch")
    con.execute(
        "CREATE INDEX IF NOT EXISTS processing_order ON docs(baseline DESC, tokens DESC, content_hash, doc_id)"
    )
    con.commit()


class ShingleCache:
    def __init__(self, con):
        self.con, self.cache, self.count = con, OrderedDict(), 0

    def get(self, doc_id):
        if doc_id in self.cache:
            self.cache.move_to_end(doc_id)
            return self.cache[doc_id]
        compressed, expected = self.con.execute(
            "SELECT text_zlib,shingle_count FROM docs WHERE doc_id=?", (doc_id,)
        ).fetchone()
        values = core.shingles(zlib.decompress(compressed).decode("utf-8"))
        if len(values) != expected:
            raise ValueError("Stored shingle count mismatch")
        # Performance/memory bound only: uncached documents are still compared.
        while self.cache and self.count + len(values) > 500_000:
            _, removed = self.cache.popitem(last=False)
            self.count -= len(removed)
        if len(values) <= 500_000:
            self.cache[doc_id] = values
            self.count += len(values)
        return values


def choose(con, row, cache):
    identifier, baseline, _, _, digest, count, signature = row
    if baseline:
        return True, "baseline", identifier, count, count, 0
    exact = con.execute("SELECT doc_id FROM exact WHERE hash=?", (digest,)).fetchone()
    if exact:
        return False, "exact", exact[0], count, count, 0
    bucket_values = core.buckets(signature)
    query = " UNION ALL ".join(
        "SELECT doc_id FROM bands WHERE band=? AND bucket=?" for _ in bucket_values
    )
    candidates = con.execute(
        f"SELECT DISTINCT d.doc_id FROM ({query}) hits JOIN docs d ON d.doc_id=hits.doc_id "
        "WHERE d.shingle_count*10 >= ?*9 AND d.shingle_count*9 <= ?*10 "
        "ORDER BY d.baseline DESC,d.tokens DESC,d.content_hash,d.doc_id",
        [v for pair in bucket_values for v in pair] + [count, count],
    )
    own, comparisons = None, 0
    for (other,) in candidates:
        if own is None:
            own = cache.get(identifier)
        intersection, union = core.similarity(own, cache.get(other))
        comparisons += 1
        if core.qualifies(intersection, union):
            return False, "near", other, intersection, union, comparisons
    return True, "unique", identifier, count, count, comparisons


def resolve(con, plan):
    cache = ShingleCache(con)
    rows = con.execute(
        "SELECT doc_id,baseline,source,tokens,content_hash,shingle_count,signature FROM docs d "
        "WHERE NOT EXISTS (SELECT 1 FROM decisions s WHERE s.doc_id=d.doc_id) "
        "ORDER BY baseline DESC,tokens DESC,content_hash,doc_id"
    )
    done = con.execute("SELECT count(*) FROM decisions").fetchone()[0]
    total = sum(f["records"] for f in plan["files"])
    for row in rows:
        keep, reason, rep, intersection, union, comparisons = choose(con, row, cache)
        identifier, _, _, _, digest, _, sig = row
        if keep:
            con.execute(
                "INSERT OR IGNORE INTO exact VALUES (?,?)", (digest, identifier)
            )
            con.executemany(
                "INSERT INTO bands VALUES (?,?,?)",
                [(band, bucket, identifier) for band, bucket in core.buckets(sig)],
            )
        con.execute(
            "INSERT INTO decisions VALUES (?,?,?,?,?,?,?)",
            (identifier, int(keep), reason, rep, intersection, union, comparisons),
        )
        done += 1
        if done % 100 == 0:
            con.commit()
        if done % 5000 == 0:
            LOG.info("Resolved %s/%s documents", done, total)
            _write_json(
                Path(plan["output"]) / "progress.json",
                {
                    "stage": "resolve",
                    "documents": done,
                    "total_documents": total,
                    "complete": False,
                },
            )
    con.commit()
    invalid = con.execute(
        "SELECT count(*) FROM decisions a LEFT JOIN decisions b ON b.doc_id=a.rep "
        "WHERE b.keep IS NULL OR b.keep<>1 OR (a.keep=0 AND a.intersection*10<a.union_size*9)"
    ).fetchone()[0]
    baseline_lost = con.execute(
        "SELECT count(*) FROM docs d JOIN decisions s USING(doc_id) WHERE d.baseline=1 AND s.keep<>1"
    ).fetchone()[0]
    if (
        invalid
        or baseline_lost
        or con.execute("SELECT count(*) FROM decisions").fetchone()[0] != total
    ):
        raise ValueError("Decision coverage/representative invariant failed")


def export_file(con, plan, digest, i):
    root, file = Path(plan["output"]), plan["files"][i]
    marker = root / "export" / f"{i:05d}.json"
    if marker.exists():
        old = read(marker)
        if old["plan_sha256"] != digest or any(
            _sha256(root / name) != sha for name, sha in old["files"].items()
        ):
            raise ValueError("Export checkpoint changed")
        return old
    decisions = con.execute(
        "SELECT d.doc_id,d.source,d.content_hash,d.tokens,s.keep,s.reason,s.rep,r.source,r.content_hash,"
        "s.intersection,s.union_size,d.origin,r.origin FROM docs d JOIN decisions s USING(doc_id) "
        "JOIN docs r ON r.doc_id=s.rep WHERE d.doc_id>=? AND d.doc_id<? ORDER BY d.doc_id",
        (i * STRIDE, (i + 1) * STRIDE),
    ).fetchall()
    if len(decisions) != file["records"]:
        raise ValueError("Incomplete export decisions")
    kept, metadata, sidecars = [], [], []
    record_rows = pq.ParquetFile(file["path"]).read().to_pylist()
    meta_rows = pq.ParquetFile(file["metadata_path"]).read().to_pylist()
    if len(record_rows) != len(decisions) or len(meta_rows) != len(decisions):
        raise ValueError("Export input coverage mismatch")
    counts = defaultdict(Counter)
    for ordinal, (record, meta, decision) in enumerate(
        zip(record_rows, meta_rows, decisions)
    ):
        (
            identifier,
            source,
            sha,
            tokens,
            keep,
            reason,
            rep,
            rep_source,
            rep_sha,
            intersection,
            union,
            origin,
            rep_origin,
        ) = decision
        if (
            identifier != i * STRIDE + ordinal
            or record["content_hash"] != sha
            or meta["content_hash"] != sha
            or record["content_length"] != tokens
        ):
            raise ValueError("Export row identity mismatch")
        counts["retained" if keep else reason].update(records=1, tokens=tokens)
        if keep:
            kept.append(record)
            metadata.append(meta)
        sidecars.append(
            {
                "doc_id": identifier,
                "source": source,
                "content_hash": sha,
                "tokens": tokens,
                "keep": bool(keep),
                "reason": reason,
                "representative_id": rep,
                "representative_source": rep_source,
                "representative_hash": rep_sha,
                "intersection": intersection,
                "union": union,
                "jaccard": intersection / union,
                "origin": origin,
                "representative_origin": rep_origin,
                "plan_sha256": digest,
            }
        )
    paths = []
    target = root / "decisions" / f"{i:05d}.parquet"
    write_parquet(target, sidecars, DECISION_SCHEMA)
    paths.append(target)
    if kept:
        for folder, records, original in (
            ("records", kept, file["path"]),
            ("metadata", metadata, file["metadata_path"]),
        ):
            target = root / folder / file["source"] / f"{i:05d}.parquet"
            write_parquet(target, records, pq.ParquetFile(original).schema_arrow)
            paths.append(target)
    report = {
        "plan_sha256": digest,
        "source": file["source"],
        "input_index": i,
        "counts": dict(counts),
        "files": {str(p.relative_to(root)): _sha256(p) for p in paths},
    }
    _write_json(marker, report)
    return report


def case(con, identifier):
    source, digest, tokens, compressed, origin = con.execute(
        "SELECT source,content_hash,tokens,text_zlib,origin FROM docs WHERE doc_id=?",
        (identifier,),
    ).fetchone()
    return {
        "doc_id": identifier,
        "source": source,
        "content_hash": digest,
        "content_length": tokens,
        "text": zlib.decompress(compressed).decode("utf-8"),
        "origin": json.loads(origin),
    }


def audit_samples(con, plan, digest):
    cases, draws, pairs = {}, [], []

    def remember(identifier):
        if identifier not in cases:
            cases[identifier] = case(con, identifier)

    sources = sorted(plan["selected_by_source"])
    for source in sources:
        total = con.execute(
            "SELECT coalesce(sum(d.tokens),0) FROM docs d JOIN decisions s USING(doc_id) WHERE d.baseline=0 AND d.source=? AND s.keep=1",
            (source,),
        ).fetchone()[0]
        rng = random.Random(f"retained:{digest}:{source}")
        offsets = (
            sorted((rng.randrange(total), draw) for draw in range(64)) if total else []
        )
        position, next_draw = 0, 0
        for identifier, tokens in con.execute(
            "SELECT d.doc_id,d.tokens FROM docs d JOIN decisions s USING(doc_id) WHERE d.baseline=0 AND d.source=? AND s.keep=1 ORDER BY d.doc_id",
            (source,),
        ):
            while (
                next_draw < len(offsets) and offsets[next_draw][0] < position + tokens
            ):
                offset, draw = offsets[next_draw]
                remember(identifier)
                draws.append(
                    {
                        "source": source,
                        "draw": draw,
                        "doc_id": identifier,
                        "token_offset": offset - position,
                    }
                )
                next_draw += 1
            position += tokens
        if next_draw != len(offsets) or position != total:
            raise ValueError("Retained sample coverage mismatch")
        for reason in ("exact", "near"):
            count = con.execute(
                "SELECT count(*) FROM docs d JOIN decisions s USING(doc_id) WHERE d.baseline=0 AND d.source=? AND s.reason=?",
                (source, reason),
            ).fetchone()[0]
            rng = random.Random(f"pairs:{digest}:{source}:{reason}")
            selected = set(rng.sample(range(count), min(16, count)))
            for ordinal, row in enumerate(
                con.execute(
                    "SELECT d.doc_id,s.rep,s.intersection,s.union_size FROM docs d JOIN decisions s USING(doc_id) WHERE d.baseline=0 AND d.source=? AND s.reason=? ORDER BY d.doc_id",
                    (source, reason),
                )
            ):
                if ordinal not in selected:
                    continue
                identifier, rep, intersection, union = row
                remember(identifier)
                remember(rep)
                actual = core.similarity(
                    core.shingles(cases[identifier]["text"]),
                    core.shingles(cases[rep]["text"]),
                )
                if actual != (intersection, union) or not core.qualifies(*actual):
                    raise ValueError("Removed-pair audit mismatch")
                pairs.append(
                    {
                        "doc_id": identifier,
                        "representative_id": rep,
                        "source": source,
                        "reason": reason,
                        "intersection": intersection,
                        "union": union,
                        "jaccard": intersection / union,
                    }
                )
    return {
        "cases": list(cases.values()),
        "token_draws": draws,
        "removed_pairs": pairs,
        "method": "64 uniform token draws with replacement per retained addition source; up to 16 uniform removed-document pairs without replacement per source/removal reason. Baseline is not resampled here.",
    }


def finish(con, plan, digest):
    root = Path(plan["output"])
    (root / "export").mkdir(exist_ok=True)
    (root / "review").mkdir(exist_ok=True)
    reports, totals = [], defaultdict(lambda: defaultdict(Counter))
    for i, file in enumerate(plan["files"]):
        if file["baseline"]:
            continue
        report = export_file(con, plan, digest, i)
        reports.append(report)
        for route, counts in report["counts"].items():
            totals[file["source"]][route].update(counts)
        if i % 500 == 0:
            LOG.info("Exported input file %s/%s", i, len(plan["files"]))
            _write_json(
                root / "progress.json",
                {
                    "stage": "export",
                    "input_file": i,
                    "total_files": len(plan["files"]),
                    "complete": False,
                },
            )
    expected_files = {name for report in reports for name in report["files"]}
    actual_files = {
        str(path.relative_to(root))
        for folder in ("records", "metadata", "decisions")
        for path in (root / folder).rglob("*.parquet")
    }
    if actual_files != expected_files:
        raise ValueError("Unexpected or missing exported Parquet files")
    for source, counts in plan["selected_by_source"].items():
        for field in ("records", "tokens"):
            if sum(c[field] for c in totals[source].values()) != counts[field]:
                raise ValueError("Export totals mismatch")
    baseline = {
        "records": plan["references"]["baseline_records"],
        "tokens": plan["references"]["baseline_tokens"],
        "files": [f for f in plan["files"] if f["baseline"]],
        "unchanged": True,
    }
    _write_json(root / "baseline-reference.json", baseline)
    _write_json(root / "review/files.json", reports)
    _write_json(root / "review/sample.json", audit_samples(con, plan, digest))
    retained = {
        field: sum(groups["retained"][field] for groups in totals.values())
        for field in ("records", "tokens")
    }
    return {
        "complete": True,
        "release_ready": False,
        "plan_sha256": digest,
        "policy": core.POLICY,
        "by_source": {s: dict(g) for s, g in totals.items()},
        "baseline": {k: baseline[k] for k in ("records", "tokens", "unchanged")},
        "retained_additions": retained,
        "baseline_plus_additions": {k: retained[k] + baseline[k] for k in retained},
        "verified_near_comparisons": con.execute(
            "SELECT coalesce(sum(comparisons),0) FROM decisions"
        ).fetchone()[0],
        "files": {
            str(p.relative_to(root)): _sha256(p)
            for p in (
                root / "baseline-reference.json",
                root / "review/files.json",
                root / "review/sample.json",
            )
        },
        "scope": "Approved web additions only; baseline preserved. Exact-unique and direct verified >=0.90 shingle-Jaccard representative selection. MinHash retrieval is approximate and can miss near duplicates. No source rewriting, quality certification, YouTube inclusion or publication approval.",
    }


def execute(project, selected, output, workers=7, references=None):
    if not 1 <= workers <= 32:
        raise ValueError("Use 1-32 signature workers")
    output = Path(output).resolve()
    for protected in (Path(project).resolve(), Path(selected).resolve()):
        if (
            output == protected
            or output.is_relative_to(protected)
            or protected.is_relative_to(output)
        ):
            raise ValueError("Keep dedup output separate from inputs")
    output.mkdir(parents=True, exist_ok=True)
    with _directory_lock(output):
        plan = make_plan(project, selected, output, references)
        digest = _sha256(output / "plan.json")
        _write_json(
            output / "progress.json", {"stage": "verify_inputs", "complete": False}
        )
        verify_inputs(plan)
        con = None
        try:
            _write_json(
                output / "progress.json", {"stage": "signatures", "complete": False}
            )
            signature_stage(plan, digest, workers)
            _write_json(
                output / "progress.json", {"stage": "load_index", "complete": False}
            )
            con = database(output, digest)
            load_signatures(con, plan, digest)
            _write_json(
                output / "progress.json", {"stage": "resolve", "complete": False}
            )
            resolve(con, plan)
            _write_json(
                output / "progress.json", {"stage": "export", "complete": False}
            )
            summary = finish(con, plan, digest)
            _write_json(
                output / "progress.json",
                {"stage": "final_verification", "complete": False},
            )
            verify_inputs(plan)
            if con.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise ValueError("Final SQLite integrity check failed")
            _write_json(output / "summary.json", summary)
            _write_json(output / "review/summary.json", summary)
            _write_json(
                output / "progress.json",
                {
                    "stage": "complete",
                    "complete": True,
                    "retained_additions": summary["retained_additions"],
                },
            )
            return summary
        finally:
            if con is not None:
                con.close()  # Uncommitted representative decisions roll back together.


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check-env", action="store_true")
    p.add_argument(
        "--prepare",
        action="store_true",
        help="Bind manifests without scanning bulk data",
    )
    p.add_argument("--project", type=Path)
    p.add_argument("--selected", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--workers", type=int, default=7)
    args = p.parse_args()
    if args.check_env:
        print("Near-dedup CPU environment OK: " + json.dumps(runtime()))
        return 0
    if any(v is None for v in (args.project, args.selected, args.output)):
        p.error("Supply --project, --selected and --output")
    if args.prepare:
        output = args.output.resolve()
        for protected in (args.project.resolve(), args.selected.resolve()):
            if (
                output == protected
                or output.is_relative_to(protected)
                or protected.is_relative_to(output)
            ):
                p.error("Keep dedup output separate from inputs")
        output.mkdir(parents=True, exist_ok=True)
        with _directory_lock(output):
            plan = make_plan(args.project, args.selected, output)
        print(
            f"Bound {len(plan['files'])} input files; baseline preserved; verified Jaccard >=0.90 required"
        )
        return 0
    started = time.monotonic()
    result = execute(args.project, args.selected, args.output, args.workers)
    LOG.info(
        "COMPLETE in %.1fs: %s",
        time.monotonic() - started,
        result["retained_additions"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

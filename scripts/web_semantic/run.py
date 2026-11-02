"""Resumable CPU web scoring and explicit, recoverable candidate selection."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
import multiprocessing
from pathlib import Path
import random
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn

from ingest.utils import compute_content_hash
from scripts.web_semantic import model
from scripts.web_filter.run import write_parquet
from scripts.youtube.download import _directory_lock, _write_json
from scripts.youtube.profile import _sha256

HERE = Path(__file__).resolve().parent
SOURCES = ("primus-fineweb", "redsage-cfw")
MODEL = None
PLAN = None
PLAN_SHA = None
SCORES = pa.schema(
    [
        ("source", pa.string()),
        ("source_shard", pa.string()),
        ("source_row", pa.int64()),
        ("dataset_revision", pa.string()),
        ("content_hash", pa.string()),
        ("content_length", pa.int64()),
        ("url", pa.string()),
        ("upstream_license", pa.string()),
        ("upstream_relevant", pa.string()),
        ("domain_score", pa.float64()),
        ("substance_score", pa.float64()),
        ("usability_score", pa.float64()),
        ("joint_score", pa.float64()),
        ("score_band", pa.int64()),
        ("route", pa.string()),
        ("model_sha256", pa.string()),
        ("plan_sha256", pa.string()),
        ("selection_decision", pa.bool_()),
    ]
)


def read_json(path):
    return json.loads(path.read_text())


def checked_path(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Path leaves bound source directory")
    return path


def code_hashes():
    return {
        p.name: _sha256(p)
        for p in sorted(HERE.iterdir())
        if p.suffix in (".py", ".sh", ".sbatch", ".txt")
    }


def make_plan(web, weights, output):
    web, weights, output = (p.resolve() for p in (web, weights, output))
    for protected in (web, weights):
        if (
            output == protected
            or output.is_relative_to(protected)
            or protected.is_relative_to(output)
        ):
            raise ValueError("Keep outputs separate from inputs/model")
    model.load_model(weights)
    parts, summaries = [], {}
    for source in SOURCES:
        root = web / source / "review"
        summary = read_json(root / "summary.json")
        if summary["complete"] is not True or summary["source"] != source:
            raise ValueError("CPU source report incomplete/wrong source")
        report_path = root / "part-reports.json"
        if _sha256(report_path) != summary["files"][report_path.name]:
            raise ValueError("CPU part inventory checksum mismatch")
        reports = read_json(report_path)
        if (
            len(reports) != summary["parts"]
            or sum(p["candidate_tokens"] for p in reports)
            != summary["candidate_tokens"]
        ):
            raise ValueError("CPU source totals mismatch")
        if len({p["output_part"] for p in reports}) != len(reports):
            raise ValueError("Duplicate CPU part")
        for part in reports:
            if (
                not part["complete"]
                or part["plan_sha256"] != summary["plan_sha256"]
                or Path(part["output_part"]).parts[0] != source
            ):
                raise ValueError("CPU part/source binding mismatch")
            parts.append({"source": source, **part})
        summaries[source] = {
            "sha256": _sha256(root / "summary.json"),
            "report_sha256": _sha256(report_path),
            "candidate_tokens": summary["candidate_tokens"],
        }
    plan = {
        "version": "web-semantic-v1",
        "web": str(web),
        "model": str(weights),
        "output": str(output),
        "model_sha256": _sha256(weights / "model.json"),
        "code": code_hashes(),
        "parents": summaries,
        "parts": parts,
        "scope": "Primus candidates and RedSage candidates explicitly labeled relevant=true. Unlabeled RedSage is deferred by researcher authorization on 2026-09-16; no original is deleted. Scores are weak-supervision estimates; final inclusion needs an explicit cutoff and review.",
        "audit_draws_per_band": 32,
    }
    output.mkdir(parents=True, exist_ok=True)
    with _directory_lock(output):
        path = output / "plan.json"
        if path.exists() and read_json(path) != plan:
            raise ValueError("Plan changed; preserve outputs and use a new directory")
        _write_json(path, plan)
    return plan


def load_plan(path):
    plan = read_json(path)
    if plan["code"] != code_hashes() or plan["model_sha256"] != _sha256(
        Path(plan["model"]) / "model.json"
    ):
        raise ValueError("Frozen code/model changed")
    model.load_model(plan["model"])
    for source, binding in plan["parents"].items():
        root = Path(plan["web"]) / source / "review"
        if (
            _sha256(root / "summary.json") != binding["sha256"]
            or _sha256(root / "part-reports.json") != binding["report_sha256"]
        ):
            raise ValueError("CPU input review changed")
    return plan


def initialize(plan, plan_sha):
    global MODEL, PLAN, PLAN_SHA
    PLAN, PLAN_SHA = plan, plan_sha
    MODEL = model.load_model(plan["model"])
    pa.set_cpu_count(1)


def work(index):
    return process_part(PLAN, PLAN_SHA, index)


def source_rows(plan, part):
    folder = checked_path(Path(plan["web"]), part["output_part"])
    for name in ("features.parquet", "candidates.parquet"):
        if _sha256(folder / name) != part["files"][name]:
            raise ValueError("CPU candidate/feature checksum mismatch")
    features = {
        r["content_hash"]: r
        for r in pq.ParquetFile(folder / "features.parquet").read().to_pylist()
        if r["route"] == "candidate"
    }
    if len(features) != part["candidate_records"]:
        raise ValueError("Feature candidate coverage mismatch")
    seen, tokens = set(), 0
    with pq.ParquetFile(folder / "candidates.parquet") as f:
        for batch in f.iter_batches(batch_size=128, use_threads=False):
            for row in batch.to_pylist():
                feature = features[row["content_hash"]]
                if (
                    row["content_hash"] in seen
                    or compute_content_hash(row["text"]) != row["content_hash"]
                ):
                    raise ValueError("Duplicate/mismatched candidate content")
                if (
                    any(
                        row[k] != feature[k]
                        for k in (
                            "source",
                            "source_shard",
                            "source_row",
                            "dataset_revision",
                            "content_hash",
                            "content_length",
                        )
                    )
                    or row["source"] != part["source"]
                ):
                    raise ValueError("Candidate/metadata correspondence mismatch")
                seen.add(row["content_hash"])
                tokens += row["content_length"]
                yield row, feature
    if len(seen) != part["candidate_records"] or tokens != part["candidate_tokens"]:
        raise ValueError("Candidate input coverage mismatch")


def part_folder(plan, index):
    return Path(plan["output"]) / "parts" / f"{index:05d}"


def cached_part(plan, plan_sha, index):
    folder = part_folder(plan, index)
    if not (folder / "summary.json").exists():
        return None
    result = read_json(folder / "summary.json")
    if (
        not result["complete"]
        or result["plan_sha256"] != plan_sha
        or result["index"] != index
        or result["scores_sha256"] != _sha256(folder / "scores.parquet")
    ):
        raise ValueError("Invalid semantic checkpoint; preserve and inspect")
    return result


def process_part(plan, plan_sha, index):
    cached = cached_part(plan, plan_sha, index)
    if cached:
        return cached
    start = time.monotonic()
    part = plan["parts"][index]
    rows, pending, texts = [], [], []
    for row, feature in source_rows(plan, part):
        deferred = (
            row["source"] == "redsage-cfw" and feature["upstream_relevant"] != "true"
        )
        result = {
            k: feature[k]
            for k in (
                "source",
                "source_shard",
                "source_row",
                "dataset_revision",
                "content_hash",
                "content_length",
                "url",
                "upstream_license",
                "upstream_relevant",
            )
        }
        result.update(
            route="deferred_upstream_scope" if deferred else "scored",
            model_sha256=plan["model_sha256"],
            plan_sha256=plan_sha,
            selection_decision=None,
        )
        rows.append(result)
        if not deferred:
            pending.append(result)
            texts.append(row["text"])
    if texts:
        # Whole bounded preparation parts; no head-only approximation/truncation.
        predictions = model.predict(texts, MODEL)
        for row, scores in zip(pending, predictions):
            row.update(
                {
                    f"{name}_score": float(score)
                    for name, score in zip(model.TARGETS, scores)
                }
            )
            row["joint_score"] = min(row["domain_score"], row["substance_score"])
            # Equal-width audit bins, not eligibility cutoffs. Usability is
            # diagnostic only: held-out negative recall was too weak to gate it.
            row["score_band"] = min(4, int(row["joint_score"] * 5))
    folder = part_folder(plan, index)
    folder.mkdir(parents=True, exist_ok=True)
    write_parquet(folder / "scores.parquet", rows, SCORES)
    groups = Counter()
    for row in rows:
        for field, value in (("records", 1), ("tokens", row["content_length"])):
            groups[(row["route"], row.get("score_band"), field)] += value
    result = {
        "complete": True,
        "plan_sha256": plan_sha,
        "index": index,
        "source": part["source"],
        "records": len(rows),
        "tokens": sum(r["content_length"] for r in rows),
        "groups": [
            {
                "route": route,
                "score_band": band,
                "records": groups[(route, band, "records")],
                "tokens": groups[(route, band, "tokens")],
            }
            for route, band in sorted({(k[0], k[1]) for k in groups}, key=str)
        ],
        "scores_sha256": _sha256(folder / "scores.parquet"),
        "seconds": time.monotonic() - start,
    }
    _write_json(folder / "summary.json", result)
    return result


def audit(plan, plan_sha, results):
    populations = defaultdict(list)
    for result in results:
        for group in result["groups"]:
            if group["route"] == "scored" and group["tokens"]:
                populations[(result["source"], group["score_band"])].append(
                    (result["index"], group["tokens"])
                )
    selected = defaultdict(list)
    # Uniform source-token locations with replacement, independently per band.
    for key, parts in sorted(populations.items()):
        total = sum(t for _, t in parts)
        rng = random.Random(f"{plan_sha}:{key}")
        for draw in range(plan["audit_draws_per_band"]):
            offset = rng.randrange(total)
            for index, tokens in parts:
                if offset < tokens:
                    selected[index].append(
                        {
                            "source": key[0],
                            "score_band": key[1],
                            "draw": draw,
                            "offset": offset,
                        }
                    )
                    break
                offset -= tokens
    draws, cases = [], {}
    for index, selections in sorted(selected.items()):
        scores = (
            pq.ParquetFile(part_folder(plan, index) / "scores.parquet")
            .read()
            .to_pylist()
        )
        wanted = set()
        for selection in selections:
            offset = selection["offset"]
            for row in scores:
                if (
                    row["route"] != "scored"
                    or row["score_band"] != selection["score_band"]
                ):
                    continue
                if offset < row["content_length"]:
                    draws.append(
                        {
                            **{k: v for k, v in selection.items() if k != "offset"},
                            "content_hash": row["content_hash"],
                            "token_offset": offset,
                            "scores": {
                                k: row[k]
                                for k in (
                                    "domain_score",
                                    "substance_score",
                                    "usability_score",
                                    "joint_score",
                                )
                            },
                        }
                    )
                    wanted.add(row["content_hash"])
                    break
                offset -= row["content_length"]
            else:
                raise ValueError("Audit token location not found")
        for row, feature in source_rows(plan, plan["parts"][index]):
            if row["content_hash"] in wanted:
                cases[row["content_hash"]] = {
                    **row,
                    "url": feature["url"],
                    "upstream_license": feature["upstream_license"],
                }
    out = Path(plan["output"]) / "review"
    out.mkdir(exist_ok=True)
    _write_json(out / "sample.json", {"cases": list(cases.values()), "draws": draws})
    _write_json(out / "parts.json", results)
    totals = defaultdict(Counter)
    bands = defaultdict(Counter)
    for r in results:
        for g in r["groups"]:
            for field in ("records", "tokens"):
                totals[r["source"]][f"{g['route']}_{field}"] += g[field]
                if g["route"] == "scored":
                    bands[(r["source"], g["score_band"])][field] += g[field]
    summary = {
        "complete": True,
        "plan_sha256": plan_sha,
        "model_sha256": plan["model_sha256"],
        "parts": len(results),
        "by_source": dict(totals),
        "score_bands": [
            {"source": k[0], "band": k[1], **v} for k, v in sorted(bands.items())
        ],
        "files": {n: _sha256(out / n) for n in ("sample.json", "parts.json")},
        "release_ready": False,
        "scope": "All in-scope CPU candidates scored without a GPU. Deferred RedSage remains recoverable. Scores are not calibrated confidence. New token-weighted samples must be reviewed. No final selection is implied by completion.",
    }
    _write_json(out / "summary.json", summary)
    return summary


def score_all(path, workers):
    if workers < 1:
        raise ValueError("Positive worker count required")
    plan, sha = load_plan(path), _sha256(path)
    with _directory_lock(Path(plan["output"])):
        runtime = {
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
        }
        runtime_path = Path(plan["output"]) / "runtime.json"
        if runtime_path.exists() and read_json(runtime_path) != runtime:
            raise ValueError("Scoring runtime changed; restore original environment")
        _write_json(runtime_path, runtime)
        results, pending = [], []
        for i in range(len(plan["parts"])):
            cached = cached_part(plan, sha, i)
            if cached:
                results.append(cached)
            else:
                pending.append(i)
        logging.info(
            "Reusing %s completed parts; %s pending", len(results), len(pending)
        )
        if workers == 1:
            initialize(plan, sha)
            for i in pending:
                results.append(process_part(plan, sha, i))
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize,
                initargs=(plan, sha),
            ) as pool:
                futures = {pool.submit(work, i): i for i in pending}
                for future in as_completed(futures):
                    results.append(future.result())
                    if len(results) % 100 == 0:
                        logging.info(
                            "%s/%s parts complete", len(results), len(plan["parts"])
                        )
        return audit(plan, sha, sorted(results, key=lambda r: r["index"]))


def selected_sample(output, files, plan):
    """Fresh uniform-token draws from the actual provisional selection."""
    selected = defaultdict(list)
    for source in SOURCES:
        source_files = [f for f in files if f["source"] == source]
        total = sum(f["tokens"] for f in source_files)
        if not total:
            continue
        rng = random.Random(f"selected:{_sha256(output / 'selection.json')}:{source}")
        for draw in range(64):
            offset = rng.randrange(total)
            for file in source_files:
                if offset < file["tokens"]:
                    selected[file["path"]].append((draw, offset))
                    break
                offset -= file["tokens"]
    cases, draws = {}, []
    by_path = {f["path"]: f for f in files}
    for name, locations in sorted(selected.items()):
        file = by_path[name]
        rows = pq.ParquetFile(output / name).read().to_pylist()
        metadata = {
            r["content_hash"]: r
            for r in pq.ParquetFile(output / file["metadata_path"]).read().to_pylist()
        }
        scores = {
            r["content_hash"]: r
            for r in pq.ParquetFile(part_folder(plan, file["index"]) / "scores.parquet")
            .read()
            .to_pylist()
        }
        for draw, offset in locations:
            for row in rows:
                if offset < row["content_length"]:
                    digest = row["content_hash"]
                    cases[digest] = {
                        **row,
                        "url": metadata[digest]["url"],
                        "upstream_license": metadata[digest]["upstream_license"],
                        "scores": scores[digest],
                    }
                    draws.append(
                        {
                            "source": row["source"],
                            "draw": draw,
                            "content_hash": digest,
                            "token_offset": offset,
                        }
                    )
                    break
                offset -= row["content_length"]
            else:
                raise ValueError("Selected token not found")
    review = output / "review"
    review.mkdir(exist_ok=True)
    _write_json(
        review / "sample.json",
        {
            "cases": list(cases.values()),
            "draws": draws,
            "method": "64 independent uniform token draws with replacement per nonempty source. Combine sources by retained token mass.",
        },
    )
    return _sha256(review / "sample.json")


def select(path, output, cutoff):
    if not 0 < cutoff < 1:
        raise ValueError("Supply an explicit researcher-approved cutoff in (0,1)")
    plan, plan_sha = load_plan(path), _sha256(path)
    output = output.resolve()
    for protected in (Path(plan["output"]), Path(plan["web"]), Path(plan["model"])):
        if (
            output == protected
            or output.is_relative_to(protected)
            or protected.is_relative_to(output)
        ):
            raise ValueError("Keep selections separate from scoring and inputs")
    output.mkdir(parents=True, exist_ok=True)
    binding = {
        "plan_sha256": plan_sha,
        "min_domain_and_substance_score": cutoff,
        "scope": "Recoverable provisional selection; no usability-score gate. Requires token review, near-dedup and release checks.",
    }
    with _directory_lock(output):
        config = output / "selection.json"
        if config.exists() and read_json(config) != binding:
            raise ValueError("Changed selection cutoff; use a new selection directory")
        _write_json(config, binding)
        _write_json(output / "summary.json", {"complete": False, **binding})
        totals = defaultdict(Counter)
        files = []
        for i, part in enumerate(plan["parts"]):
            if not cached_part(plan, plan_sha, i):
                raise ValueError("Scoring must complete before selection")
            scores = (
                pq.ParquetFile(part_folder(plan, i) / "scores.parquet")
                .read()
                .to_pylist()
            )
            wanted = {
                r["content_hash"]
                for r in scores
                if r["route"] == "scored" and r["joint_score"] >= cutoff
            }
            if not wanted:
                continue
            kept, metadata = [], []
            for row, feature in source_rows(plan, part):
                if row["content_hash"] in wanted:
                    kept.append(row)
                    metadata.append(feature)
            if len(kept) != len(wanted):
                raise ValueError("Selection/source mismatch")
            source = part["source"]
            folder = output / "records" / source
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / f"{i:05d}.parquet"
            schema = pq.ParquetFile(
                checked_path(Path(plan["web"]), part["output_part"])
                / "candidates.parquet"
            ).schema_arrow
            write_parquet(target, kept, schema)
            metadata_target = output / "metadata" / source / f"{i:05d}.parquet"
            metadata_target.parent.mkdir(parents=True, exist_ok=True)
            metadata_schema = pq.ParquetFile(
                checked_path(Path(plan["web"]), part["output_part"])
                / "features.parquet"
            ).schema_arrow
            write_parquet(metadata_target, metadata, metadata_schema)
            totals[source].update(
                records=len(kept), tokens=sum(r["content_length"] for r in kept)
            )
            files.append(
                {
                    "path": str(target.relative_to(output)),
                    "metadata_path": str(metadata_target.relative_to(output)),
                    "source": source,
                    "index": i,
                    "records": len(kept),
                    "tokens": sum(r["content_length"] for r in kept),
                    "sha256": _sha256(target),
                    "metadata_sha256": _sha256(metadata_target),
                }
            )
        summary = {
            "complete": True,
            **binding,
            "by_source": dict(totals),
            "files": files,
            "release_ready": False,
            "review_sample_sha256": selected_sample(output, files, plan),
        }
        _write_json(output / "summary.json", summary)
        _write_json(output / "review/summary.json", summary)
        return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    create = sub.add_parser("plan")
    for name in ("web", "model", "output"):
        create.add_argument(f"--{name}", type=Path, required=True)
    score = sub.add_parser("score")
    score.add_argument("--plan", type=Path, required=True)
    score.add_argument("--workers", type=int, default=7)
    pick = sub.add_parser("select")
    pick.add_argument("--plan", type=Path, required=True)
    pick.add_argument("--output", type=Path, required=True)
    pick.add_argument("--min-score", type=float, required=True)
    sub.add_parser("check-env")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.command == "plan":
        result = make_plan(args.web, args.model, args.output)
        print(
            f"Frozen {len(result['parts'])} CPU input parts; unlabeled RedSage deferred"
        )
    elif args.command == "score":
        print(json.dumps(score_all(args.plan, args.workers), indent=2))
    elif args.command == "select":
        result = select(args.plan, args.output, args.min_score)
        print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
    else:
        if sklearn.__version__ != "1.7.2":
            raise ValueError("Install the pinned scikit-learn runtime")
        print(
            f"CPU semantic environment OK: sklearn {sklearn.__version__}, NumPy {np.__version__}"
        )


if __name__ == "__main__":
    main()

"""CPU-only web quality screening, exact baseline overlap and review sampling.

No language model, network request, keyword scope list, or content rewriting.
Prepared web inputs are already exact-unique across both web sources. Existing
source selection is evidence, not an assertion of final semantic quality.
"""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
import multiprocessing
from pathlib import Path
import random
import re
import tarfile

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from ingest.utils import compute_content_hash
from scripts.curation import prepare, score_partitions
from scripts.youtube.download import _directory_lock, _write_json
from scripts.youtube.profile import _sha256
from scripts.web_corpora import profile as profiler

HERE = Path(__file__).resolve().parent
SOURCES = ("primus-fineweb", "redsage-cfw")
VERSION = "web-cpu-filter-v1"
BAD_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufffd]")
BASELINE_HASHES = set()
SIDECAR = pa.schema(
    [
        ("source", pa.string()),
        ("dataset_revision", pa.string()),
        ("source_shard", pa.string()),
        ("source_row", pa.int64()),
        ("content_hash", pa.string()),
        ("content_length", pa.int64()),
        ("url", pa.string()),
        ("domain", pa.string()),
        ("upstream_id", pa.string()),
        ("upstream_license", pa.string()),
        ("upstream_language", pa.string()),
        ("upstream_relevant", pa.string()),
        ("upstream_probability", pa.float64()),
        ("scope_stratum", pa.string()),
        ("characters", pa.int64()),
        ("nonblank_lines", pa.int64()),
        ("repeated_line_fraction", pa.float64()),
        ("bad_character_fraction", pa.float64()),
        ("route", pa.string()),
        ("reasons", pa.list_(pa.string())),
        ("plan_sha256", pa.string()),
        ("selection_decision", pa.bool_()),
    ]
)


def code_hashes():
    paths = [p for p in HERE.iterdir() if p.suffix in {".py", ".sh", ".sbatch"}]
    paths += [Path(m.__file__) for m in (prepare, score_partitions, profiler)]
    return {str(p.relative_to(HERE.parent.parent)): _sha256(p) for p in sorted(paths)}


def validate_policy(policy):
    if policy.get("version") != "web-cpu-quality-v1" or policy.get("mode") not in {
        "metrics_only",
        "quarantine",
    }:
        raise ValueError("Unknown web policy")
    for key in ("max_repeated_line_fraction", "max_bad_character_fraction"):
        value = policy.get(key)
        if policy["mode"] == "metrics_only" and value is not None:
            raise ValueError("Metrics-only policy cannot silently apply thresholds")
        if policy["mode"] == "quarantine" and (
            type(value) not in (int, float) or not 0 < value <= 1
        ):
            raise ValueError(
                "Quarantine requires explicit researcher-approved fractions"
            )


def quality_features(text):
    lines = Counter(line.strip() for line in text.splitlines() if line.strip())
    denominator = sum(len(line) * count for line, count in lines.items())
    repeated = sum(len(line) * (count - 1) for line, count in lines.items())
    return {
        "characters": len(text),
        "nonblank_lines": sum(lines.values()),
        "repeated_line_fraction": repeated / denominator if denominator else 0.0,
        "bad_character_fraction": len(BAD_CHARACTERS.findall(text)) / len(text)
        if text
        else 0.0,
    }


def route(features, policy, baseline_match):
    if baseline_match:
        return "baseline_duplicate", ["exact_text_already_in_baseline"]
    reasons = []
    if policy["mode"] == "quarantine":
        if features["repeated_line_fraction"] >= policy["max_repeated_line_fraction"]:
            reasons.append("heavy_repeated_lines")
        if features["bad_character_fraction"] >= policy["max_bad_character_fraction"]:
            reasons.append("replacement_or_control_characters")
    return ("quarantine" if reasons else "candidate"), reasons


def stratum(source, relevant):
    if source == "primus-fineweb":
        return "upstream_dataset_selected"
    if relevant == "true":
        return "upstream_relevant_true"
    if relevant == "false":
        return "upstream_relevant_false"
    return "upstream_unlabeled"


def make_plan(project, prepared, web_root, output, policy_path):
    project, prepared, web_root, output = (
        p.resolve() for p in (project, prepared, web_root, output)
    )
    for protected in (prepared, web_root, project / "data"):
        if (
            output == protected
            or output.is_relative_to(protected)
            or protected.is_relative_to(output)
        ):
            raise ValueError("Keep CPU outputs separate from inputs")
    policy = json.loads(policy_path.read_text())
    validate_policy(policy)
    summary = score_partitions.load_prepared(prepared)
    config = json.loads((prepared / "run-config.json").read_text())
    inputs = {
        (r["source"], r["index"]): r for r in config["inputs"] if r["source"] in SOURCES
    }
    groups = defaultdict(list)
    for part in summary["parts"]:
        components = Path(part["path"]).parts
        if components[1] in SOURCES:
            groups[(components[1], int(components[2]))].append(part)
    reports, tasks = {}, []
    for source, name in (
        ("primus-fineweb", "profile-v2"),
        ("redsage-cfw", "profile-v1"),
    ):
        path = web_root / source / name / "summary.json"
        if _sha256(path) != config["source_summaries"][source]:
            raise ValueError("Profile differs from prepared input provenance")
        report = json.loads(path.read_text())
        if not report["complete"]:
            raise ValueError("Profile incomplete")
        reports[str(path)] = _sha256(path)
        for (src, index), parts in sorted(groups.items()):
            if src != source:
                continue
            item, shard = inputs[(src, index)], report["shards"][index]
            metadata = (
                web_root / src / name / "metadata" / f"{index:05d}.parquet"
            ).resolve()
            if (
                shard["source_shard"] != item["source_shard"]
                or str(metadata) != item["metadata"]
            ):
                raise ValueError("Metadata locator differs from preparation")
            tasks.append(
                {
                    "source": src,
                    "index": index,
                    "source_shard": item["source_shard"],
                    "dataset_revision": item["dataset_revision"],
                    "metadata": str(metadata),
                    "metadata_sha256": shard["metadata_sha256"],
                    "parts": sorted(parts, key=lambda p: p["path"]),
                }
            )
    baseline_config = project / "config/retained_baseline.json"
    baseline = json.loads(baseline_config.read_text())
    paths = []
    for relative in baseline["inputs"]:
        path = (project / relative).resolve()
        if not path.is_relative_to(project / "data"):
            raise ValueError("Baseline path escapes data directory")
        found = (
            [path]
            if path.is_file()
            else sorted(
                p for p in path.rglob("*.parquet") if not p.name.startswith("._")
            )
        )
        if not found:
            raise ValueError(f"Missing baseline input: {relative}")
        paths.extend(str(p) for p in found)
    plan = {
        "version": VERSION,
        "prepared": str(prepared),
        "output": str(output),
        "prepared_summary_sha256": _sha256(prepared / "summary.json"),
        "lineage_sha256": summary["index"]["lineage_sha256"],
        "profile_summaries": reports,
        "baseline_config": str(baseline_config),
        "baseline_config_sha256": _sha256(baseline_config),
        "baseline_files": sorted(set(paths)),
        "baseline_expected_records": baseline["expected_records"],
        "tasks": tasks,
        "policy": policy,
        "code": code_hashes(),
        "versions": {"duckdb": duckdb.__version__, "pyarrow": pa.__version__},
        "scope": "CPU quality candidates, not individually LLM-classified additions. Upstream relevance strata preserved. No near-deduplication, source rewriting, or public release approval.",
    }
    output.mkdir(parents=True, exist_ok=True)
    with _directory_lock(output):
        path = output / "plan.json"
        if path.exists() and json.loads(path.read_text()) != plan:
            raise ValueError(
                "CPU plan changed; preserve outputs and use another directory"
            )
        _write_json(path, plan)
    return plan


def load_plan(path):
    plan = json.loads(path.read_text())
    if (
        plan["version"] != VERSION
        or plan["code"] != code_hashes()
        or path.resolve() != Path(plan["output"]) / "plan.json"
        or plan["versions"] != {"duckdb": duckdb.__version__, "pyarrow": pa.__version__}
    ):
        raise ValueError("CPU code/environment/plan binding changed")
    if (
        _sha256(Path(plan["prepared"]) / "summary.json")
        != plan["prepared_summary_sha256"]
    ):
        raise ValueError("Prepared input changed")
    if _sha256(Path(plan["baseline_config"])) != plan["baseline_config_sha256"]:
        raise ValueError("Baseline selection changed")
    for name, checksum in plan["profile_summaries"].items():
        if _sha256(Path(name)) != checksum:
            raise ValueError("Source profile changed")
    validate_policy(plan["policy"])
    return plan


def baseline_index(path):
    plan = load_plan(path)
    output = Path(plan["output"])
    (output / "baseline-lock").mkdir(exist_ok=True)
    with _directory_lock(output / "baseline-lock"):
        binding = {name: _sha256(Path(name)) for name in plan["baseline_files"]}
        report = output / "baseline-summary.json"
        target = output / "baseline-hashes.parquet"
        if report.exists():
            previous = json.loads(report.read_text())
            if (
                previous["plan_sha256"] != _sha256(path)
                or previous["input_files"] != binding
                or previous["sha256"] != _sha256(target)
            ):
                raise ValueError("Baseline checkpoint changed")
            return previous
        with duckdb.connect(config={"memory_limit": "8GB", "threads": 4}) as con:
            con.read_parquet(
                plan["baseline_files"], union_by_name=True, hive_partitioning=False
            ).create_view("baseline")
            records = con.execute("SELECT count(*) FROM baseline").fetchone()[0]
            invalid = con.execute(
                "SELECT count(*) FROM baseline WHERE content IS NULL OR trim(content)='' OR content_hash IS NULL OR sha256(content)<>content_hash"
            ).fetchone()[0]
            if invalid or records != plan["baseline_expected_records"]:
                raise ValueError(
                    "Baseline contents/count differ from retained inventory"
                )
            temporary = target.with_suffix(".tmp")
            con.sql(
                "SELECT DISTINCT sha256(content) AS content_hash FROM baseline ORDER BY content_hash"
            ).write_parquet(str(temporary), compression="zstd")
            temporary.replace(target)
        if binding != {name: _sha256(Path(name)) for name in plan["baseline_files"]}:
            raise ValueError("Baseline changed during indexing")
        result = {
            "complete": True,
            "records": records,
            "input_files": binding,
            "plan_sha256": _sha256(path),
            "sha256": _sha256(target),
        }
        _write_json(report, result)
        return result


def initialize_worker(baseline):
    global BASELINE_HASHES
    pa.set_cpu_count(1)
    BASELINE_HASHES = set(
        pq.ParquetFile(baseline).read(columns=["content_hash"]).column(0).to_pylist()
    )


def write_parquet(path, rows, schema):
    temporary = path.with_suffix(".tmp")
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd"
    )
    temporary.replace(path)


def process_group(task, prepared, output, policy, plan_sha):
    metadata_path = Path(task["metadata"])
    if _sha256(metadata_path) != task["metadata_sha256"]:
        raise ValueError("Profile metadata checksum mismatch")
    metadata = {}
    with pq.ParquetFile(metadata_path) as f:
        for batch in f.iter_batches(batch_size=2048, use_threads=False):
            for row in batch.to_pylist():
                key = (row["source_shard"], row["source_row"])
                if key in metadata or row["source"] != task["source"]:
                    raise ValueError("Invalid metadata key/source")
                metadata[key] = row
    results = []
    for part in task["parts"]:
        source_path = (prepared / part["path"]).resolve()
        if (
            not source_path.is_relative_to(prepared)
            or _sha256(source_path) != part["sha256"]
        ):
            raise ValueError("Prepared part checksum/path mismatch")
        folder = (
            output
            / task["source"]
            / "parts"
            / f"{task['index']:05d}"
            / source_path.stem
        )
        folder.mkdir(parents=True, exist_ok=True)
        checkpoint = folder / "summary.json"
        if checkpoint.exists():
            cached = json.loads(checkpoint.read_text())
            if (
                cached["plan_sha256"] != plan_sha
                or cached["input_sha256"] != part["sha256"]
            ):
                raise ValueError("Part checkpoint belongs to another run")
            if (
                cached.get("complete") is True
                and set(cached["files"]) == {"features.parquet", "candidates.parquet"}
                and all(
                    _sha256(folder / name) == digest
                    for name, digest in cached["files"].items()
                )
            ):
                results.append(cached)
                continue
            raise ValueError("Part output corruption; preserve and inspect")
        kept, features, populations = [], [], defaultdict(lambda: [0, 0])
        with pq.ParquetFile(source_path) as f:
            for batch in f.iter_batches(batch_size=128, use_threads=False):
                for row in batch.to_pylist():
                    meta = metadata[(row["source_shard"], row["source_row"])]
                    if (
                        row["source"] != task["source"]
                        or row["kind"] != "web"
                        or row["dataset_revision"] != task["dataset_revision"]
                        or row["source_shard"] != task["source_shard"]
                        or not row["text"].strip()
                        or row["content_length"] <= 0
                        or meta["content_hash"] != row["content_hash"]
                        or meta["content_length"] != row["content_length"]
                        or compute_content_hash(row["text"]) != row["content_hash"]
                    ):
                        raise ValueError(
                            "Source content/metadata correspondence failed"
                        )
                    quality = quality_features(row["text"])
                    action, reasons = route(
                        quality, policy, row["content_hash"] in BASELINE_HASHES
                    )
                    group = stratum(row["source"], meta["upstream_relevant"])
                    features.append(
                        {
                            **{
                                k: row[k]
                                for k in (
                                    "source",
                                    "dataset_revision",
                                    "source_shard",
                                    "source_row",
                                    "content_hash",
                                    "content_length",
                                )
                            },
                            **{
                                k: meta[k]
                                for k in (
                                    "url",
                                    "domain",
                                    "upstream_id",
                                    "upstream_license",
                                    "upstream_relevant",
                                    "upstream_probability",
                                )
                            },
                            "upstream_language": meta["language"],
                            "scope_stratum": group,
                            **quality,
                            "route": action,
                            "reasons": reasons,
                            "plan_sha256": plan_sha,
                            "selection_decision": None,
                        }
                    )
                    populations[(group, action)][0] += 1
                    populations[(group, action)][1] += row["content_length"]
                    if action == "candidate":
                        kept.append(row)
        if (
            len(features) != part["records"]
            or sum(r["content_length"] for r in features) != part["tokens"]
        ):
            raise ValueError("Output coverage differs from prepared part")
        write_parquet(folder / "candidates.parquet", kept, prepare.SCHEMA)
        write_parquet(folder / "features.parquet", features, SIDECAR)
        result = {
            "complete": True,
            "plan_sha256": plan_sha,
            "input_path": part["path"],
            "input_sha256": part["sha256"],
            "output_part": str(folder.relative_to(output)),
            "records": len(features),
            "tokens": part["tokens"],
            "candidate_records": len(kept),
            "candidate_tokens": sum(r["content_length"] for r in kept),
            "strata": [
                {"scope_stratum": k[0], "route": k[1], "records": v[0], "tokens": v[1]}
                for k, v in sorted(populations.items())
            ],
            "files": {
                n: _sha256(folder / n)
                for n in ("features.parquet", "candidates.parquet")
            },
        }
        _write_json(checkpoint, result)
        results.append(result)
    return results


def review_bundle(plan, plan_sha, source, results):
    root = Path(plan["output"])
    out = root / source / "review"
    out.mkdir(exist_ok=True)
    populations = defaultdict(list)
    for result in sorted(results, key=lambda r: r["input_path"]):
        for group in result["strata"]:
            populations[(group["scope_stratum"], group["route"])].append(
                (result, group)
            )
    chosen = defaultdict(list)
    strata = []
    for key, parts in sorted(populations.items()):
        ends, total = [], 0
        for _, group in parts:
            total += group["tokens"]
            ends.append(total)
        generator = random.Random(f"{plan_sha}:{source}:{key}")
        # Audit workload count, not a content threshold.
        for draw in range(64):
            position = generator.randrange(total)
            i = bisect_right(ends, position)
            chosen[parts[i][0]["input_path"]].append(
                {
                    "stratum": key,
                    "draw": draw,
                    "part_token_offset": position - (ends[i - 1] if i else 0),
                }
            )
        strata.append(
            {
                "scope_stratum": key[0],
                "route": key[1],
                "tokens": total,
                "records": sum(g["records"] for _, g in parts),
                "draws": 64,
            }
        )
    by_path = {r["input_path"]: r for r in results}
    cases, draws = {}, []
    for name, samples in sorted(chosen.items()):
        result = by_path[name]
        sidecars = (
            pq.ParquetFile(root / result["output_part"] / "features.parquet")
            .read()
            .to_pylist()
        )
        wanted = {}
        for selection in samples:
            key = selection["stratum"]
            offset = selection["part_token_offset"]
            for row in sidecars:
                if (row["scope_stratum"], row["route"]) != key:
                    continue
                if offset < row["content_length"]:
                    cid = f"{source}:{row['source_shard']}:{row['source_row']}"
                    wanted[row["source_row"]] = (cid, row)
                    draws.append(
                        {
                            "source": source,
                            "scope_stratum": key[0],
                            "route": key[1],
                            "draw": selection["draw"],
                            "case_id": cid,
                            "token_offset": offset,
                        }
                    )
                    break
                offset -= row["content_length"]
            else:
                raise ValueError("Sample token offset exceeds sidecar population")
        with pq.ParquetFile(Path(plan["prepared"]) / name) as f:
            for batch in f.iter_batches(batch_size=128, use_threads=False):
                for row in batch.to_pylist():
                    if row["source_row"] in wanted:
                        cid, feature = wanted.pop(row["source_row"])
                        if row["content_hash"] != feature["content_hash"]:
                            raise ValueError("Sample text differs from sidecar")
                        cases[cid] = {
                            "case_id": cid,
                            **row,
                            "quality_features": feature,
                        }
        if wanted:
            raise ValueError("Sample records missing from source")
    names = ("review-packet.json", "token-draws.json", "part-reports.json")
    _write_json(
        out / names[0],
        {
            "cases": list(cases.values()),
            "scope": "Original full texts with features; no independent semantic labels supplied.",
        },
    )
    _write_json(
        out / names[1],
        {
            "strata": strata,
            "draws": draws,
            "scope": "Uniform cl100k_base token locations with replacement within each source/upstream-label/route stratum. Preserve repeated draws and combine strata by retained token mass.",
        },
    )
    _write_json(out / names[2], results)
    report = {
        "complete": True,
        "source": source,
        "plan_sha256": plan_sha,
        **{
            k: sum(r[k] for r in results)
            for k in ("records", "tokens", "candidate_records", "candidate_tokens")
        },
        "strata": strata,
        "parts": len(results),
        "sampled_documents": len(cases),
        "inference_performed": False,
        "semantic_quality_validated": False,
        "release_ready": False,
        "policy": plan["policy"],
        "files": {n: _sha256(out / n) for n in names},
        "scope": plan["scope"],
    }
    _write_json(out / "summary.json", report)
    temp = out / "review-bundle.tar.gz.partial"
    with tarfile.open(temp, "w:gz") as archive:
        for name in (*names, "summary.json"):
            archive.add(out / name, arcname=name)
    bundle = out / "review-bundle.tar.gz"
    temp.replace(bundle)
    _write_json(
        out / "bundle-summary.json",
        {"bytes": bundle.stat().st_size, "sha256": _sha256(bundle)},
    )
    return report


def process_source(path, source, workers):
    plan = load_plan(path)
    plan_sha = _sha256(path)
    root = Path(plan["output"])
    destination = root / source
    destination.mkdir(exist_ok=True)
    with _directory_lock(destination):
        _write_json(
            destination / "summary.json", {"complete": False, "status": "processing"}
        )
        review = destination / "review"
        if review.exists():
            _write_json(
                review / "summary.json", {"complete": False, "status": "processing"}
            )
            (review / "bundle-summary.json").unlink(missing_ok=True)
        index = json.loads((root / "baseline-summary.json").read_text())
        baseline = root / "baseline-hashes.parquet"
        if index["plan_sha256"] != plan_sha or index["sha256"] != _sha256(baseline):
            raise ValueError("Baseline index binding changed")
        if (
            _sha256(Path(plan["prepared"]) / "lineage.parquet")
            != plan["lineage_sha256"]
        ):
            raise ValueError("Prepared alias lineage changed")
        tasks = [t for t in plan["tasks"] if t["source"] == source]
        if not tasks:
            raise ValueError("No prepared partitions for requested source")
        results = []
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_worker,
            initargs=(baseline,),
        ) as pool:
            futures = [
                pool.submit(
                    process_group,
                    task,
                    Path(plan["prepared"]),
                    root,
                    plan["policy"],
                    plan_sha,
                )
                for task in tasks
            ]
            for n, future in enumerate(as_completed(futures), 1):
                results.extend(future.result())
                logging.info("%s: completed %s/%s source shards", source, n, len(tasks))
        if len(results) != sum(len(t["parts"]) for t in tasks):
            raise ValueError("Source partition coverage mismatch")
        report = review_bundle(plan, plan_sha, source, results)
        _write_json(destination / "summary.json", report)
        return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    create = sub.add_parser("plan")
    for name in ("project", "prepared", "web-root", "output"):
        create.add_argument("--" + name, type=Path, required=True)
    create.add_argument("--policy", type=Path, default=HERE / "policy.json")
    for name in ("index", "source"):
        child = sub.add_parser(name)
        child.add_argument("--plan", type=Path, required=True)
        if name == "source":
            child.add_argument("--source", choices=SOURCES, required=True)
            child.add_argument("--workers", type=int, default=7)
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.command == "plan":
        plan = make_plan(
            args.project, args.prepared, args.web_root, args.output, args.policy
        )
        print(
            f"CPU plan ready: {len(plan['tasks'])} source shards; mode={plan['policy']['mode']}"
        )
        return 0
    if args.command == "index":
        baseline_index(args.plan)
        print("Baseline hash index ready")
    else:
        if args.workers < 1:
            p.error("workers must be positive")
        report = process_source(args.plan, args.source, args.workers)
        print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

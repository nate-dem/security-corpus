"""Finalize audited YouTube candidates against the already retained corpus.

Uses the approved dedup engine unchanged. Baseline and retained web signatures
are checksum-checked and reused; only new YouTube signatures are computed.
Outputs preserve prepared text/lineage schemas and never modify source files.
"""

import argparse
from collections import Counter
import importlib.metadata
import json
import logging
from pathlib import Path
import zlib

import pyarrow.parquet as pq

from ingest.utils import compute_content_hash, compute_token_count
from scripts.curation_audit import run as audit
from scripts.curation_single import run as single
from scripts.near_dedup import core, run as dedup
from scripts.youtube.download import _directory_lock, _write_json
from scripts.youtube.profile import _sha256

LOG = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
SOURCE = "youtube-commons"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def checked_files(root, files):
    for name, digest in files.items():
        require(
            _sha256(dedup.checked(root, name)) == digest, f"Changed artifact: {name}"
        )


def completed_web(web):
    plan, summary = read(web / "plan.json"), read(web / "summary.json")
    digest = _sha256(web / "plan.json")
    require(
        summary.get("complete") is True
        and summary == read(web / "review/summary.json")
        and summary["plan_sha256"] == digest,
        "Web deduplication is incomplete or changed",
    )
    require(
        plan["policy"] == core.POLICY
        and summary["policy"] == core.POLICY
        and plan["code"] == dedup.code_hashes()
        and plan["runtime"] == dedup.runtime(),
        "Web signature code/runtime/policy changed",
    )
    checked_files(web, summary["files"])
    reports = read(web / "review/files.json")
    by_index = {r["input_index"]: r for r in reports}
    expected = {i for i, f in enumerate(plan["files"]) if not f["baseline"]}
    require(
        len(by_index) == len(reports) and set(by_index) == expected,
        "Web export reports have incomplete coverage",
    )
    files = []
    for index, original in enumerate(plan["files"]):
        if original["baseline"]:
            files.append({**original, "prior_index": index})
            continue
        report = by_index[index]
        require(
            report["plan_sha256"] == digest and report["source"] == original["source"],
            "Web export plan/source mismatch",
        )
        checked_files(web, report["files"])
        for field in ("records", "tokens"):
            require(
                sum(c[field] for c in report["counts"].values()) == original[field],
                "Web export/input count mismatch",
            )
        retained = report["counts"].get("retained", {"records": 0, "tokens": 0})
        if not retained["records"]:
            continue
        record = f"records/{original['source']}/{index:05d}.parquet"
        meta = f"metadata/{original['source']}/{index:05d}.parquet"
        decisions = f"decisions/{index:05d}.parquet"
        files.append(
            {
                "path": str(dedup.checked(web, record)),
                "sha256": report["files"][record],
                "metadata_path": str(dedup.checked(web, meta)),
                "metadata_sha256": report["files"][meta],
                "source": original["source"],
                "baseline": True,
                **retained,
                "prior_index": index,
                "prior_decisions": str(dedup.checked(web, decisions)),
                "prior_decisions_sha256": report["files"][decisions],
            }
        )
    require(
        {k: sum(f[k] for f in files) for k in ("records", "tokens")}
        == summary["baseline_plus_additions"],
        "Retained baseline/web totals disagree",
    )
    return plan, files


def audited_candidates(output):
    imported = read(output / "import-audit.json")
    audit_root = output / "audit"
    report = read(audit_root / "review/summary.json")
    require(
        imported["complete"]
        and report["audit_complete"]
        and _sha256(audit_root / "review/summary.json")
        == imported["audit_summary_sha256"],
        "Combined import/audit must complete before selection",
    )
    path = Path(imported["plan_path"])
    require(
        _sha256(path) == imported["plan_sha256"] == report["plan_sha256"],
        "Scoring plan changed after audit",
    )
    plan = single.load_plan(path)
    checked_files(audit_root / "review", report["files"])
    frozen = read(audit_root / "frozen.json")
    require(
        frozen == read(audit_root / "review/frozen.json")
        and frozen["auditor_code"] == audit.auditor_code()
        and {p["slot"] for p in frozen["parts"]} == set(imported["completed_slots"]),
        "Audited subset changed",
    )
    parts = read(audit_root / "review/part-audits.json")
    require(
        len(parts) == len(frozen["parts"])
        and {p["slot"] for p in parts} == set(imported["completed_slots"]),
        "Audited part coverage differs from import",
    )
    by_slot = {p["slot"]: p for p in parts}
    files = []
    for entry in frozen["parts"]:
        slot = entry["slot"]
        part = by_slot[slot]
        require(
            part["snapshot_sha256"] == _sha256(audit_root / "frozen.json")
            and audit.fingerprint(plan, entry) == part["input_fingerprint"],
            f"Audited worker/source files changed: {slot}",
        )
        candidate = single.folder(plan, slot) / "candidates.parquet"
        worker = read(candidate.parent / "summary.json")
        require(
            _sha256(candidate) == worker["files"]["candidates.parquet"],
            f"Candidate export changed: {slot}",
        )
        require(
            pq.ParquetFile(candidate).metadata.num_rows == part["candidate_records"],
            f"Candidate row count differs from audit: {slot}",
        )
        if not part["candidate_records"]:
            continue
        # Prepared rows already carry source/revision/shard/row/hash lineage.
        # Reuse that schema as metadata; source alias lineage remains bound below.
        files.append(
            {
                "path": str(candidate),
                "sha256": _sha256(candidate),
                "metadata_path": str(candidate),
                "metadata_sha256": _sha256(candidate),
                "source": SOURCE,
                "baseline": False,
                "slot": slot,
                "records": part["candidate_records"],
                "tokens": part["candidate_tokens"],
            }
        )
    counts = report["by_source"][SOURCE]
    require(
        all(
            sum(f[k] for f in files) == counts[f"candidate_{k}"]
            for k in ("records", "tokens")
        ),
        "Candidate totals differ from audit",
    )
    lineage = Path(plan["input_dir"]) / "lineage.parquet"
    require(_sha256(lineage) == plan["lineage_sha256"], "Source alias lineage changed")
    return report, files, {"path": str(lineage), "sha256": plan["lineage_sha256"]}


def make_plan(output, web):
    web_plan, baseline = completed_web(web)
    report, additions, lineage = audited_candidates(output)
    counts = report["by_source"][SOURCE]
    plan = {
        "version": "youtube-finalized-v1",
        "output": str(output / "dedup"),
        "policy": core.POLICY,
        "runtime": dedup.runtime(),
        "code": dedup.code_hashes(),
        "finalizer_code": {p.name: _sha256(p) for p in sorted(HERE.glob("*.py"))},
        "tokenizer_version": importlib.metadata.version("tiktoken"),
        "selection": "Researcher requested inclusion of existing eligible YouTube candidates; retain original routing, then apply approved deduplication.",
        "precedence": "Preserve existing baseline and retained web; YouTube additions use the approved longer-document/direct-representative rule.",
        "files": baseline + additions,
        "references": {
            "baseline_records": sum(f["records"] for f in baseline),
            "baseline_tokens": sum(f["tokens"] for f in baseline),
        },
        "selected_by_source": {
            SOURCE: {k: counts[f"candidate_{k}"] for k in ("records", "tokens")}
        },
        "web": str(web),
        "web_plan_sha256": _sha256(web / "plan.json"),
        "web_summary_sha256": _sha256(web / "summary.json"),
        "import_audit_sha256": _sha256(output / "import-audit.json"),
        "audit_summary_sha256": _sha256(output / "audit/review/summary.json"),
        "source_alias_lineage": lineage,
        "completed_parts": report["completed_parts"],
        "planned_parts": report["planned_parts"],
        "unprocessed_slots": report["unprocessed_slots"],
        "all_planned_work_complete": report["all_planned_work_complete"],
        "unresolved_documents": counts["unresolved_documents"],
        "signature_batch_size": dedup.BATCH,
    }
    target = output / "dedup/plan.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    require(
        not target.exists() or read(target) == plan,
        "Frozen finalization inputs/runtime/code changed; restore before resuming",
    )
    _write_json(target, plan)
    return plan, web_plan


def reused_rows(web, web_plan, file):
    """Read prior verified signatures, keeping only the prior retained copies."""
    index = file["prior_index"]
    original = web_plan["files"][index]
    decisions = None
    if not original["baseline"]:
        require(
            _sha256(Path(file["prior_decisions"])) == file["prior_decisions_sha256"],
            "Prior web decisions changed",
        )
        rows = pq.ParquetFile(file["prior_decisions"]).read().to_pylist()
        require(
            [r["doc_id"] for r in rows]
            == list(
                range(index * dedup.STRIDE, index * dedup.STRIDE + original["records"])
            ),
            "Prior decision coverage changed",
        )
        decisions = {r["doc_id"]: r for r in rows}
    digest = _sha256(web / "plan.json")
    for start in range(0, original["records"], dedup.BATCH):
        path = dedup.location(web, index, start)
        count = min(dedup.BATCH, original["records"] - start)
        require(
            dedup.cached_signature(path, digest, count) is not None,
            "Original web signature cache is missing; preserve the completed web run",
        )
        rows = pq.ParquetFile(path).read().to_pylist()
        require(
            [r["doc_id"] for r in rows]
            == list(
                range(
                    index * dedup.STRIDE + start, index * dedup.STRIDE + start + count
                )
            ),
            "Prior signature identity changed",
        )
        for row in rows:
            if decisions is not None:
                decision = decisions[row["doc_id"]]
                require(
                    all(
                        row[k] == decision[k]
                        for k in ("source", "tokens", "content_hash")
                    ),
                    "Prior signature/decision mismatch",
                )
                if not decision["keep"]:
                    continue
            yield row


def copy_signatures(plan, web_plan, digest):
    """Rebind already verified baseline/web sketches without rerunning MinHash."""
    root, web = Path(plan["output"]), Path(plan["web"])
    for index, file in enumerate(plan["files"]):
        if not file["baseline"]:
            continue
        cached = [
            dedup.cached_signature(
                dedup.location(root, index, start),
                digest,
                min(dedup.BATCH, file["records"] - start),
            )
            for start in range(0, file["records"], dedup.BATCH)
        ]
        if all(cached):
            require(
                sum(c["tokens"] for c in cached) == file["tokens"],
                "Reused token totals changed",
            )
            continue
        originals = iter(reused_rows(web, web_plan, file))
        count, tokens = 0, 0
        # Compare cached text/hash/count identities against the checksum-bound
        # retained Parquet. No baseline or web tokenizer/model pass is repeated.
        for batch in pq.ParquetFile(file["path"]).iter_batches(batch_size=dedup.BATCH):
            copied = []
            for record in batch.to_pylist():
                row = next(originals, None)
                text = record["content"] if "content" in record else record["text"]
                require(
                    row is not None
                    and row["content_hash"] == record["content_hash"]
                    and row["tokens"] == record["content_length"]
                    and zlib.decompress(row["text_zlib"]).decode("utf-8") == text,
                    "Prior signature/retained text mismatch",
                )
                copied.append(
                    {**row, "doc_id": index * dedup.STRIDE + count + len(copied)}
                )
            target = dedup.location(root, index, count)
            if dedup.cached_signature(target, digest, len(copied)) is None:
                dedup.write_parquet(target, copied, dedup.SKETCH_SCHEMA)
                _write_json(
                    target.with_suffix(".json"),
                    {
                        "plan_sha256": digest,
                        "sha256": _sha256(target),
                        "records": len(copied),
                        "tokens": sum(r["tokens"] for r in copied),
                    },
                )
            count += len(copied)
            tokens += sum(r["tokens"] for r in copied)
        require(
            next(originals, None) is None
            and (count, tokens) == (file["records"], file["tokens"]),
            "Reused signature count mismatch",
        )
        if index % 100 == 0:
            LOG.info("Reused retained-corpus signature file %s", index)


def recount(root, summary):
    """Count the actual exported YouTube bytes, with hash/token recomputation."""
    totals = Counter(records=0, tokens=0)
    files = []
    reports = read(root / "review/files.json")
    for report in reports:
        checked_files(root, report["files"])
        for name, digest in report["files"].items():
            if not name.startswith("records/"):
                continue
            counts = Counter(records=0, tokens=0)
            for batch in pq.ParquetFile(root / name).iter_batches(batch_size=128):
                for row in batch.to_pylist():
                    tokens = compute_token_count(row["text"])
                    require(
                        row["source"] == SOURCE
                        and compute_content_hash(row["text"]) == row["content_hash"]
                        and tokens == row["content_length"],
                        "Final YouTube text/hash/token mismatch",
                    )
                    counts.update(records=1, tokens=tokens)
            require(
                dict(counts) == report["counts"]["retained"],
                "Exported token totals disagree",
            )
            totals.update(counts)
            files.append({"path": str(root / name), "sha256": digest, **counts})
    require(
        dict(totals) == summary["retained_additions"], "Final YouTube totals disagree"
    )
    return files


def execute(output, web, workers=7):
    output, web = Path(output).resolve(), Path(web).resolve()
    require(1 <= workers <= 32, "Use 1-32 signature workers")
    for protected in (web, HERE.parents[1]):
        require(
            output != protected
            and not output.is_relative_to(protected)
            and not protected.is_relative_to(output),
            "Keep output separate from web and code",
        )
    output.mkdir(parents=True, exist_ok=True)
    with _directory_lock(output):
        progress = output / "progress.json"
        _write_json(output / "summary.json", {"complete": False})
        _write_json(progress, {"stage": "verify_audited_inputs", "complete": False})
        plan, web_plan = make_plan(output, web)
        root, con = Path(plan["output"]), None
        digest = _sha256(root / "plan.json")
        try:
            dedup.verify_inputs(plan)
            _write_json(
                progress, {"stage": "reuse_baseline_web_signatures", "complete": False}
            )
            copy_signatures(plan, web_plan, digest)
            _write_json(progress, {"stage": "youtube_signatures", "complete": False})
            dedup.signature_stage(plan, digest, workers)
            con = dedup.database(root, digest)
            dedup.load_signatures(con, plan, digest)
            _write_json(progress, {"stage": "deduplicate", "complete": False})
            dedup.resolve(con, plan)
            summary = dedup.finish(con, plan, digest)
            summary["scope"] = (
                "YouTube eligible subset against preserved baseline and retained web. Direct verified >=0.90 shingle-Jaccard matches; approximate retrieval can miss near duplicates. Full-source YouTube filtering remains separately reported."
            )
            _write_json(
                progress, {"stage": "recount_exported_tokens", "complete": False}
            )
            youtube_files = recount(root, summary)
            dedup.verify_inputs(plan)
            require(
                con.execute("PRAGMA quick_check").fetchall() == [("ok",)],
                "Final SQLite check failed",
            )
            _write_json(root / "summary.json", summary)
            _write_json(root / "review/summary.json", summary)
            # A file manifest composes the existing corpus and new subset without
            # copying 2.46B already-retained tokens or changing their schemas.
            retained_files = [
                {k: f[k] for k in ("path", "sha256", "records", "tokens")}
                for f in plan["files"]
                if f["baseline"]
            ]
            _write_json(
                output / "corpus-manifest.json",
                {
                    "files": retained_files + youtube_files,
                    "totals": summary["baseline_plus_additions"],
                    "youtube_source_alias_lineage": plan["source_alias_lineage"],
                    "dedup_plan_sha256": digest,
                    "scope": "Existing baseline and web plus finalized YouTube subset; publication checks remain separate.",
                },
            )
            result = {
                "complete": True,
                "release_ready": False,
                "quality_review_complete": False,
                "scope": "Completed YouTube subset finalized; does not claim exhaustive filtering or public release readiness.",
                "tokenizer": "cl100k_base",
                "completed_parts": plan["completed_parts"],
                "planned_parts": plan["planned_parts"],
                "unprocessed_slots": plan["unprocessed_slots"],
                "all_planned_work_complete": plan["all_planned_work_complete"],
                "unresolved_documents": plan["unresolved_documents"],
                "baseline_and_web": summary["baseline"],
                "youtube_before_dedup": plan["selected_by_source"][SOURCE],
                "youtube_after_dedup": summary["retained_additions"],
                "youtube_routes": summary["by_source"].get(SOURCE, {}),
                "corpus_total": summary["baseline_plus_additions"],
                "corpus_manifest_sha256": _sha256(output / "corpus-manifest.json"),
                "dedup_plan_sha256": digest,
                "dedup_summary_sha256": _sha256(root / "summary.json"),
                "audit_summary_sha256": plan["audit_summary_sha256"],
            }
            _write_json(output / "summary.json", result)
            _write_json(
                progress,
                {
                    "stage": "complete",
                    "complete": True,
                    "corpus_total": result["corpus_total"],
                },
            )
            LOG.info(
                "FINALIZED: %s YouTube tokens; %s corpus tokens",
                result["youtube_after_dedup"]["tokens"],
                result["corpus_total"]["tokens"],
            )
            return result
        except Exception as exc:
            _write_json(
                progress, {"stage": "failed", "complete": False, "error": str(exc)}
            )
            raise
        finally:
            if con is not None:
                con.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--web", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    print(json.dumps(execute(args.output, args.web, args.workers), indent=2))


if __name__ == "__main__":
    main()

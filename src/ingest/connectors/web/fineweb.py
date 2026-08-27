"""FineWeb security-relevance filtering helpers.

This module implements a small DSIR-style log-ratio scorer. It intentionally
uses only lightweight local dependencies so fixture tests and smoke runs do not
require downloading FineWeb.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import math
import pickle
import re
import statistics
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pyarrow.parquet as pq

from ingest.connectors.base import NormalizedData, WebDocumentData
from ingest.utils import compute_content_hash, compute_token_count


FINEWEB_SOURCE_ID = "fineweb-security"
FINEWEB_LICENSE = "FineWeb Terms"
SCORER_VERSION = "dsir_log_ratio_v1"
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+.#-]{1,}", re.IGNORECASE)


class FineWebInputError(RuntimeError):
    """Raised when FineWeb input is missing or malformed."""


@dataclass
class DsirScorer:
    """A compact DSIR-style scorer using smoothed domain/background log ratios."""

    weights: dict[str, float]
    default_weight: float
    metadata: dict[str, Any] = field(default_factory=dict)
    ngram_range: tuple[int, int] = (1, 2)

    def score(self, text: str) -> float:
        features = tokenize_features(text, self.ngram_range)
        if not features:
            return float("-inf")
        total = sum(features.values())
        weighted = sum(count * self.weights.get(token, self.default_weight) for token, count in features.items())
        return weighted / max(total, 1)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "DsirScorer":
        with path.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a DsirScorer")
        return obj


def tokenize_features(text: str, ngram_range: tuple[int, int] = (1, 2), *, max_terms: int = 20_000) -> Counter[str]:
    terms = [match.group(0).lower() for match in TOKEN_RE.finditer(text or "")]
    if len(terms) > max_terms:
        terms = terms[:max_terms]
    counts: Counter[str] = Counter()
    min_n, max_n = ngram_range
    for n in range(min_n, max_n + 1):
        if n <= 0 or len(terms) < n:
            continue
        if n == 1:
            counts.update(terms)
        else:
            counts.update("_".join(terms[idx : idx + n]) for idx in range(len(terms) - n + 1))
    return counts


def fit_dsir_scorer(
    positive_texts: Iterable[str],
    background_texts: Iterable[str],
    *,
    positive_limit: int | None = None,
    background_limit: int | None = None,
    ngram_range: tuple[int, int] = (1, 2),
    min_feature_count: int = 2,
    alpha: float = 1.0,
    metadata: dict[str, Any] | None = None,
) -> DsirScorer:
    positive_counts: Counter[str] = Counter()
    background_counts: Counter[str] = Counter()
    positive_docs = 0
    background_docs = 0

    for text in positive_texts:
        if positive_limit is not None and positive_docs >= positive_limit:
            break
        features = tokenize_features(text, ngram_range)
        if features:
            positive_counts.update(features)
            positive_docs += 1

    for text in background_texts:
        if background_limit is not None and background_docs >= background_limit:
            break
        features = tokenize_features(text, ngram_range)
        if features:
            background_counts.update(features)
            background_docs += 1

    if not positive_counts:
        raise ValueError("no positive text features found")
    if not background_counts:
        raise ValueError("no background text features found")

    vocab = {term for term, count in (positive_counts + background_counts).items() if count >= min_feature_count}
    if not vocab:
        raise ValueError("no features met min_feature_count")

    positive_total = sum(positive_counts.values())
    background_total = sum(background_counts.values())
    vocab_size = len(vocab)
    weights: dict[str, float] = {}
    for term in vocab:
        p_domain = (positive_counts[term] + alpha) / (positive_total + alpha * vocab_size)
        p_background = (background_counts[term] + alpha) / (background_total + alpha * vocab_size)
        weights[term] = math.log(p_domain / p_background)

    default_weight = math.log((alpha / (positive_total + alpha * vocab_size)) / (alpha / (background_total + alpha * vocab_size)))
    scorer_metadata = {
        "version": SCORER_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "positive_docs": positive_docs,
        "background_docs": background_docs,
        "positive_features": positive_total,
        "background_features": background_total,
        "vocab_size": vocab_size,
        "ngram_range": list(ngram_range),
        "min_feature_count": min_feature_count,
        "alpha": alpha,
        "git_commit": _git_commit(),
    }
    if metadata:
        scorer_metadata.update(metadata)
    return DsirScorer(weights=weights, default_weight=default_weight, metadata=scorer_metadata, ngram_range=ngram_range)


def normalize_fineweb_record(record: dict[str, Any], *, score: float | None = None) -> WebDocumentData:
    content = fineweb_record_text(record)
    if not content:
        raise FineWebInputError("FineWeb record has no text/content")
    source_record_id = _record_id(record, content)
    title = _first_str(record, "title", "metadata.title")
    source_url = _first_str(record, "url", "source_url", "metadata.url")
    if title:
        title = " ".join(title.split())
    return WebDocumentData(
        source_id=FINEWEB_SOURCE_ID,
        source_record_id=source_record_id,
        record_id=f"{FINEWEB_SOURCE_ID}:{source_record_id}",
        title=title,
        content=content,
        content_length=compute_token_count(content),
        content_hash=compute_content_hash(content),
        ingested_at=datetime.now(timezone.utc),
        published_at=None,
        source_url=source_url,
        license=FINEWEB_LICENSE,
        raw=None,
        dsir_score=score,
        language=_first_str(record, "language", "lang", "metadata.language"),
    )


def docs_from_input(
    *,
    input_glob: str | None = None,
    fineweb_dataset: str = "HuggingFaceFW/fineweb",
    fineweb_config: str | None = None,
    split: str = "train",
    streaming: bool = True,
) -> Iterator[dict[str, Any]]:
    if input_glob:
        paths = sorted(glob.glob(input_glob))
        if not paths:
            raise FineWebInputError(f"input glob matched no files: {input_glob}")
        for path in paths:
            yield from _docs_from_file(Path(path))
        return

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise FineWebInputError("Hugging Face input requires `pip install datasets` or use --input-glob") from exc
    kwargs: dict[str, Any] = {"split": split, "streaming": streaming}
    dataset = load_dataset(fineweb_dataset, fineweb_config, **kwargs) if fineweb_config else load_dataset(fineweb_dataset, **kwargs)
    for record in dataset:
        yield dict(record)


def fineweb_record_text(record: dict[str, Any]) -> str:
    return _record_text(record)


def iter_positive_texts(globs: Sequence[str], *, limit: int | None = None) -> Iterator[str]:
    yielded = 0
    for pattern in globs:
        for file_path in sorted(glob.glob(pattern)):
            table = pq.ParquetFile(file_path).read(columns=["content"])
            for value in table.column("content").to_pylist():
                if isinstance(value, str) and value.strip():
                    yield value
                    yielded += 1
                    if limit is not None and yielded >= limit:
                        return


def write_fineweb_records(
    records: Iterable[NormalizedData],
    output_dir: Path,
    *,
    shard_name: str,
    overwrite: bool = False,
) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_path = output_dir / f"source_id={FINEWEB_SOURCE_ID}" / f"{shard_name}.parquet"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; use --overwrite to replace it")
    rows = [record.model_dump() for record in records]
    if not rows:
        return 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output_path, compression="snappy")
    return len(rows)


def audit_fineweb_output(output_dir: Path, report_dir: Path) -> dict[str, Any]:
    files = sorted((output_dir / f"source_id={FINEWEB_SOURCE_ID}").glob("*.parquet"))
    rows = 0
    tokens = 0
    hashes: Counter[str] = Counter()
    urls = 0
    lengths: list[int] = []
    scores: list[float] = []
    top_samples: list[dict[str, Any]] = []
    borderline_samples: list[dict[str, Any]] = []
    fallback_samples: list[dict[str, Any]] = []
    for path in files:
        table = pq.ParquetFile(path).read()
        data = table.to_pylist()
        rows += len(data)
        for row in data:
            length = int(row.get("content_length") or 0)
            tokens += length
            lengths.append(length)
            if row.get("dsir_score") is not None:
                scores.append(float(row["dsir_score"]))
            content_hash = row.get("content_hash")
            if content_hash:
                hashes[str(content_hash)] += 1
            if row.get("source_url"):
                urls += 1
            sample = _audit_sample_row(row, length)
            if sample["dsir_score"] is None:
                if len(fallback_samples) < 40:
                    fallback_samples.append(sample)
            else:
                top_samples.append(sample)
                top_samples.sort(key=lambda item: float(item["dsir_score"]), reverse=True)
                del top_samples[20:]
                borderline_samples.append(sample)
                borderline_samples.sort(key=lambda item: float(item["dsir_score"]))
                del borderline_samples[20:]
    duplicate_hashes = sum(1 for count in hashes.values() if count > 1)
    summary = {
        "source_id": FINEWEB_SOURCE_ID,
        "files": len(files),
        "rows": rows,
        "tokens": tokens,
        "duplicate_hashes": duplicate_hashes,
        "url_coverage": (urls / rows) if rows else 0.0,
        "content_length": _distribution(lengths),
        "dsir_score": _distribution(scores),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (report_dir / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    samples = []
    for sample in top_samples:
        samples.append({"sample_kind": "high_score", **sample})
    for sample in borderline_samples:
        samples.append({"sample_kind": "borderline_kept", **sample})
    if not samples:
        samples = [{"sample_kind": "sample", **sample} for sample in fallback_samples]
    with (report_dir / "audit_sample.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_kind", "dsir_score", "source_record_id", "title", "source_url", "content_length", "content_preview"],
        )
        writer.writeheader()
        writer.writerows(samples)
    return summary


def build_slurm_script(
    *,
    tasks: int,
    array_concurrency: int,
    account: str,
    partition: str,
    qos: str,
    cpus_per_task: int,
    mem: str,
    time_limit: str,
    command: str,
) -> str:
    last_task = tasks - 1
    return f"""#!/bin/bash
#SBATCH --job-name=fineweb-filter
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --array=0-{last_task}%{array_concurrency}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --output=logs/fineweb/task_%a.out
#SBATCH --error=logs/fineweb/task_%a.err

set -euo pipefail

cd "${{SLURM_SUBMIT_DIR:-$PWD}}"
{command} --task-id "$SLURM_ARRAY_TASK_ID" --tasks {tasks}
"""


def _docs_from_file(path: Path) -> Iterator[dict[str, Any]]:
    suffixes = "".join(path.suffixes)
    if path.suffix == ".parquet":
        table = pq.ParquetFile(path).read()
        for row in table.to_pylist():
            yield row
        return
    if path.suffix == ".jsonl" or suffixes.endswith(".jsonl.gz"):
        opener = _open_text(path)
        with opener as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    raise FineWebInputError(f"unsupported FineWeb input file: {path}")


def _open_text(path: Path):
    if "".join(path.suffixes).endswith(".jsonl.gz"):
        import gzip

        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _record_text(record: dict[str, Any]) -> str:
    value = _first_str(record, "text", "content", "document", "raw_content")
    return "\n".join(str(value or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()).strip()


def _record_id(record: dict[str, Any], content: str) -> str:
    explicit = _first_str(record, "id", "doc_id", "warc_record_id", "url", "source_url")
    if explicit:
        return _safe_id(explicit)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:24]


def _first_str(record: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value: Any = record
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                value = None
                break
            value = value[part]
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _safe_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    text = re.sub(r"_+", "_", text).strip("._-")
    if len(text) > 120:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
        text = f"{text[:100]}_{digest}"
    return text or hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _audit_sample_row(row: dict[str, Any], length: int) -> dict[str, Any]:
    raw_score = row.get("dsir_score")
    return {
        "dsir_score": float(raw_score) if raw_score is not None else None,
        "source_record_id": row.get("source_record_id"),
        "title": row.get("title"),
        "source_url": row.get("source_url"),
        "content_length": length,
        "content_preview": str(row.get("content") or "")[:400].replace("\n", " "),
    }


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "p50": None, "p90": None, "p99": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(values),
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p99": _percentile(ordered, 0.99),
    }


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, round((len(sorted_values) - 1) * q)))
    return float(sorted_values[idx])


def _summary_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# FineWeb Security Audit",
            "",
            f"- Source: `{summary['source_id']}`",
            f"- Parquet files: {summary['files']}",
            f"- Rows: {summary['rows']:,}",
            f"- Tokens: {summary['tokens']:,}",
            f"- Duplicate content hashes: {summary['duplicate_hashes']:,}",
            f"- URL coverage: {summary['url_coverage']:.1%}",
            f"- Content length p50/p90/p99: {summary['content_length']['p50']} / {summary['content_length']['p90']} / {summary['content_length']['p99']}",
            f"- DSIR score p50/p90/p99: {summary['dsir_score']['p50']} / {summary['dsir_score']['p90']} / {summary['dsir_score']['p99']}",
            "",
        ]
    )


def _git_commit() -> str | None:
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    except OSError:
        return None
    return proc.stdout.strip() or None

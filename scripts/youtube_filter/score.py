"""Run the transcript-content pilot with pinned Qwen/vLLM and resumable evidence.

Dry-run loads only the tokenizer. GPU inference emits labels, not a final
keep/drop policy. Missing/failed spans keep a document unresolved.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import re
import socket
import time

from ingest.utils import compute_content_hash, compute_token_count
from scripts.youtube.download import _directory_lock, _write_json
from scripts.youtube.profile import _sha256
from .rubric import MODEL, MODEL_REVISION, PROMPT_VERSION, RESPONSE_SCHEMA, parse_response
from . import rubric as quote_rubric, rubric_segments, rubric_scope


LOG = logging.getLogger("youtube-content")
TOKENIZER_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
                   "merges.txt", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja")


def load_tokenizer(model: str, revision: str, local_files_only: bool):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    # Transformers 4.57.3's Mistral regex probe queries unpinned Hub tags even
    # for Qwen when loading by repo ID. A local, pinned tokenizer snapshot with
    # config.json avoids that probe and makes offline restarts possible.
    snapshot = Path(snapshot_download(model, revision=revision, local_files_only=local_files_only,
        allow_patterns=list(TOKENIZER_FILES)))
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if not (snapshot / name).is_file():
            raise ValueError(f"Incomplete tokenizer cache ({name}); warm the pinned cache with an online dry run")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    return tokenizer, snapshot


def load_candidates(root: Path, max_documents: int) -> tuple[list[dict], str]:
    report = json.loads((root / "summary.json").read_text())
    if report.get("complete") is not True:
        raise ValueError("Pilot preparation is incomplete")
    for name in ("candidates.jsonl", "lineage.jsonl"):
        if _sha256(root / name) != report["files"][name]:
            raise ValueError(f"Pilot input checksum mismatch: {name}")
    rows, seen = [], set()
    with (root / "candidates.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            text = row["text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Pilot candidate is not nonempty text")
            digest = compute_content_hash(text)
            if (digest != row["content_hash"] or digest in seen
                    or compute_token_count(text) != row["content_length"]
                    or row["dataset_revision"] != report["dataset_revision"]):
                raise ValueError("Pilot hash, duplicate, tokenizer, or revision mismatch")
            seen.add(digest)
            rows.append(row)
            if len(rows) > max_documents:
                raise ValueError("Input exceeds --max-documents pilot budget; no records were truncated")
    if not rows or len(rows) != report["unique_candidate_texts"]:
        raise ValueError("Empty pilot or candidate count mismatch")
    return rows, _sha256(root / "summary.json")


def _versions(dry_run):
    names = ["transformers", "tokenizers", "tiktoken", "jinja2", "huggingface-hub"] + ([] if dry_run else ["vllm", "torch"])
    return {name: importlib.metadata.version(name) for name in names}


def _task_key(digest, span):
    return f"{digest}-{span['start']}-{span['end']}"


def _cached(path, task, focal, response_parser=parse_response, expected_provenance=None):
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        raise ValueError(f"Unreadable decision checkpoint: {path}; preserve it and use a new output-dir") from None
    if value.get("task_sha256") != task["task_sha256"]:
        raise ValueError("Decision checkpoint does not match this request")
    # Prompt/tokenizer equality alone cannot distinguish 8B from 32B results.
    # Audit callers may separately validate historical provenance; inference
    # callers always supply the current model, prompt, and configuration hash.
    for key, expected in (expected_provenance or {}).items():
        if value.get(key) != expected:
            raise ValueError(f"Decision checkpoint provenance mismatch ({key}): {path}; use a new output-dir")
    for key in ("content_hash", "start", "end"):
        if key in task and value.get(key) != task[key]:
            raise ValueError(f"Decision checkpoint span mismatch ({key}): {path}")
    if not value.get("attempts"):
        raise ValueError("Decision checkpoint has no attempts")
    latest = value["attempts"][-1]
    # Revalidate cached raw model text instead of trusting saved parse status.
    return {**value, **response_parser(latest["raw_response"], focal, task["start"], latest["finish_reason"])}


def score_batch(llm, sampling, tasks, documents, output, model, revision,
                response_parser=parse_response, prompt_version=PROMPT_VERSION, run_config_sha256=None):
    provenance = {"model": model, "model_revision": revision, "prompt_version": prompt_version}
    if run_config_sha256 is not None:
        provenance["run_config_sha256"] = run_config_sha256
    started = time.monotonic()
    results = llm.generate([{"prompt_token_ids": t["prompt_token_ids"]} for t in tasks],
                           sampling, use_tqdm=False)
    elapsed = time.monotonic() - started
    if len(results) != len(tasks):
        raise ValueError("Inference did not return one response for every request")
    produced = []
    for task, generated in zip(tasks, results):
        if generated.prompt_token_ids != task["prompt_token_ids"]:
            raise ValueError("Inference response does not match its prompt token IDs")
        if len(generated.outputs) != 1:
            raise ValueError("Expected one completion per span")
        completion = generated.outputs[0]
        focal = documents[task["content_hash"]]["text"][task["start"]:task["end"]]
        path = output / "decisions" / f"{task['key']}.json"
        existing = _cached(path, task, focal, response_parser, provenance)
        attempts = existing["attempts"] if existing else []
        attempt = {"raw_response": completion.text, "finish_reason": completion.finish_reason,
                   "input_tokens": len(task["prompt_token_ids"]), "output_tokens": len(completion.token_ids),
                   "created_at": datetime.now(timezone.utc).isoformat(),
                   "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "hostname": socket.gethostname()}
        result = {"task_sha256": task["task_sha256"], "content_hash": task["content_hash"],
                  "start": task["start"], "end": task["end"], **provenance,
                  "attempts": [*attempts, attempt],
                  **response_parser(completion.text, focal, task["start"], completion.finish_reason)}
        _write_json(path, result)
        produced.append(result)
    return produced, {"seconds": elapsed, "input_tokens": sum(len(t["prompt_token_ids"]) for t in tasks),
                      "output_tokens": sum(len(r.outputs[0].token_ids) for r in results)}


def report_documents(rows, tasks, output, response_parser=parse_response, expected_provenance=None):
    by_document = defaultdict(list)
    for task in tasks:
        by_document[task["content_hash"]].append(task)
    reports = []
    for row in rows:
        labels = defaultdict(Counter)
        parse_states = Counter()
        errors = Counter()
        covered = 0
        for task in by_document[row["content_hash"]]:
            focal = row["text"][task["start"]:task["end"]]
            saved = _cached(output / "decisions" / f"{task['key']}.json", task, focal, response_parser, expected_provenance)
            state = saved["parse_status"] if saved else "unprocessed"
            parse_states[state] += 1
            if state != "ok":
                errors[saved.get("error", state) if saved else state] += 1
            if state == "ok":
                covered += len(focal)
                for dimension in ("text_language", "security_relevance", "technical_substance", "text_usability", "mixed_content"):
                    labels[dimension][saved["labels"][dimension]] += len(focal)
        reports.append({"content_hash": row["content_hash"], "content_length": row["content_length"],
                        "sample_arms": row["sample_arms"], "locations": row["locations"],
                        "span_count": len(by_document[row["content_hash"]]), "parse_states": dict(parse_states),
                        "validation_errors": dict(errors),
                        "coverage_complete": covered == len(row["text"]),
                        "characters_with_valid_labels": covered, "total_characters": len(row["text"]),
                        "label_character_counts": {k: dict(v) for k, v in labels.items()},
                        "has_uncertain_labels": any("uncertain" in c for c in labels.values()),
                        "selection_decision": None})
    return reports


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--evidence-format", choices=["quotes", "segment-ids"], default="quotes")
    parser.add_argument("--rubric-version", choices=["legacy", "scope-v3"], default="legacy",
                        help="scope-v3 clarifies the approved computing scope; requires segment-ids evidence")
    parser.add_argument("--local-files-only", action="store_true", help="Tokenizer only; offline tokenizer cache check")
    parser.add_argument("--focal-tokens", type=int, default=4096)
    parser.add_argument("--context-chars", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--attempts-per-run", type=int, default=1)
    parser.add_argument("--max-documents", type=int, default=2000, help="Pilot workload guard; fails rather than truncating")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.85)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    rubric = rubric_segments if args.evidence_format == "segment-ids" else quote_rubric
    if args.rubric_version == "scope-v3":
        if args.evidence_format != "segment-ids":
            parser.error("scope-v3 requires --evidence-format segment-ids")
        rubric = rubric_scope
    if not re.fullmatch(r"[0-9a-f]{40}", args.model_revision):
        parser.error("model-revision must be an immutable commit SHA")
    for name in ("focal_tokens", "batch_size", "attempts_per_run", "max_documents", "tensor_parallel_size"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if args.context_chars < 0 or not 0 < args.max_output_tokens < args.max_model_len or not 0 < args.gpu_memory_utilization <= 1:
        parser.error("Invalid context, output, or GPU memory budget")
    root, output = args.input_dir.resolve(), args.output_dir.resolve()
    if output == root or root.is_relative_to(output) or output.is_relative_to(root):
        parser.error("Input and output directories must be separate")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output.mkdir(parents=True, exist_ok=True)
    with _directory_lock(root), _directory_lock(output):
        rows, input_digest = load_candidates(root, args.max_documents)
        tokenizer, tokenizer_snapshot = load_tokenizer(args.model, args.model_revision, args.local_files_only)
        config = {**{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  "input_summary_sha256": input_digest, "prompt_version": rubric.PROMPT_VERSION,
                  "rubric_sha256": _sha256(Path(rubric.__file__)),
                  "base_rubric_sha256": _sha256(Path(quote_rubric.__file__)),
                  "segment_rubric_sha256": _sha256(Path(rubric_segments.__file__)),
                  "runner_sha256": _sha256(Path(__file__)), "versions": _versions(args.dry_run),
                  "tokenizer_files": {name: _sha256(tokenizer_snapshot / name) for name in TOKENIZER_FILES
                                      if (tokenizer_snapshot / name).is_file()},
                  "temperature": 0.0, "enable_thinking": False, "enforce_eager": True}
        config_path = output / "run-config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError("Inference configuration changed; use a new output-dir")
        if not config_path.exists() and any((output / "decisions").glob("*.json")):
            raise ValueError("Decision checkpoints lack their run configuration; preserve them and use a new output-dir")
        _write_json(config_path, config)
        config_digest = _sha256(config_path)
        provenance = {"model": args.model, "model_revision": args.model_revision,
                      "prompt_version": rubric.PROMPT_VERSION, "run_config_sha256": config_digest}
        _write_json(output / "summary.json", {"complete": False, "status": "planning"})
        tasks = []
        documents = {row["content_hash"]: row for row in rows}
        for row in rows:
            spans = rubric.make_spans(row["text"], tokenizer, focal_tokens=args.focal_tokens,
                               context_chars=args.context_chars, max_model_len=args.max_model_len,
                               max_output_tokens=args.max_output_tokens)
            for span in spans:
                task = {**span, "content_hash": row["content_hash"], "key": _task_key(row["content_hash"], span)}
                task["task_sha256"] = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()
                tasks.append(task)
        request_path = output / "requests.jsonl"
        temporary = request_path.with_suffix(".tmp")
        with temporary.open("w") as handle:
            for task in tasks:
                # Full rendered prompt for inspection; model IDs are derivable from pinned tokenizer.
                handle.write(json.dumps({k: v for k, v in task.items() if k != "prompt_token_ids"}, ensure_ascii=False) + "\n")
        temporary.replace(request_path)
        LOG.info("Planned %d spans covering %d entire transcripts", len(tasks), len(rows))
        if args.dry_run:
            _write_json(output / "summary.json", {"complete": True, "dry_run": True,
                "inference_performed": False, "documents": len(rows), "spans": len(tasks),
                "prompt_tokens": sum(len(t["prompt_token_ids"]) for t in tasks),
                "max_prompt_tokens": max(len(t["prompt_token_ids"]) for t in tasks),
                "requests_sha256": _sha256(request_path)})
            LOG.info("DRY RUN COMPLETE: tokenizer/prompt/coverage only; no model weights or GPU used")
            return 0
        (output / "decisions").mkdir(exist_ok=True)
        pending = [t for t in tasks if not (saved := _cached(output / "decisions" / f"{t['key']}.json", t,
                   documents[t["content_hash"]]["text"][t["start"]:t["end"]], rubric.parse_response,
                   provenance)) or saved["parse_status"] != "ok"]
        cached_valid_spans = len(tasks) - len(pending)
        LOG.info("Reused %d valid spans; %d need generation", cached_valid_spans, len(pending))
        metrics = {"generation_seconds": 0.0, "input_tokens": 0, "output_tokens": 0, "attempted_spans": 0}
        model_load_seconds = 0.0
        if pending:
            from vllm import LLM, SamplingParams
            from vllm.sampling_params import StructuredOutputsParams
            start = time.monotonic()
            llm = LLM(model=args.model, revision=args.model_revision, tokenizer=str(tokenizer_snapshot),
                      tokenizer_revision=args.model_revision, trust_remote_code=False, dtype="bfloat16",
                      max_model_len=args.max_model_len, max_num_seqs=args.batch_size,
                      tensor_parallel_size=args.tensor_parallel_size,
                      gpu_memory_utilization=args.gpu_memory_utilization, enable_prefix_caching=True,
                      enforce_eager=True, seed=args.seed)
            model_load_seconds = time.monotonic() - start
            sampling = SamplingParams(temperature=0.0, max_tokens=args.max_output_tokens, seed=args.seed,
                                       structured_outputs=StructuredOutputsParams(json=RESPONSE_SCHEMA))
            for attempt in range(args.attempts_per_run):
                retry = []
                for index in range(0, len(pending), args.batch_size):
                    batch = pending[index:index+args.batch_size]
                    batch_sampling = sampling
                    if args.evidence_format == "segment-ids":
                        batch_sampling = [SamplingParams(temperature=0.0, max_tokens=args.max_output_tokens, seed=args.seed,
                            structured_outputs=StructuredOutputsParams(json=t["response_schema"])) for t in batch]
                    results, timing = score_batch(llm, batch_sampling, batch, documents, output, args.model, args.model_revision,
                                                  rubric.parse_response, rubric.PROMPT_VERSION, config_digest)
                    metrics["generation_seconds"] += timing["seconds"]
                    metrics["input_tokens"] += timing["input_tokens"]
                    metrics["output_tokens"] += timing["output_tokens"]
                    metrics["attempted_spans"] += len(batch)
                    retry.extend(t for t, r in zip(batch, results) if r["parse_status"] != "ok")
                    LOG.info("Attempt %d: %d/%d spans; %d failures so far", attempt+1,
                             min(index+len(batch), len(pending)), len(pending), len(retry))
                pending = retry
                if not pending:
                    break
        reports = report_documents(rows, tasks, output, rubric.parse_response, provenance)
        # Historical token totals survive a zero-inference restart. Wall time
        # cannot be reconstructed from timestamps, so it stays per invocation.
        saved_generation = {"attempts": 0, "input_tokens": 0, "output_tokens": 0, "slurm_job_ids": []}
        job_ids = set()
        for task in tasks:
            saved = json.loads((output / "decisions" / f"{task['key']}.json").read_text())
            for attempt in saved["attempts"]:
                saved_generation["attempts"] += 1
                saved_generation["input_tokens"] += attempt["input_tokens"]
                saved_generation["output_tokens"] += attempt["output_tokens"]
                if attempt.get("slurm_job_id"):
                    job_ids.add(attempt["slurm_job_id"])
        saved_generation["slurm_job_ids"] = sorted(job_ids)
        path = output / "document_report.jsonl"
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as handle:
            for report in reports:
                handle.write(json.dumps(report) + "\n")
        temporary.replace(path)
        complete = all(row["coverage_complete"] for row in reports)
        errors = Counter()
        for row in reports:
            errors.update(row["validation_errors"])
        _write_json(output / "summary.json", {"complete": complete, "dry_run": False,
            "inference_performed": metrics["attempted_spans"] > 0,
            "inference_performed_scope": "This invocation only; saved_generation includes all recorded attempts.",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cached_valid_spans_at_start": cached_valid_spans, "saved_generation": saved_generation,
            "run_config_sha256": config_digest, "model": args.model, "model_revision": args.model_revision,
            "documents": len(rows), "spans": len(tasks), "unresolved_spans": len(pending),
            "documents_with_uncertain_labels": sum(r["has_uncertain_labels"] for r in reports),
            "candidate_tokens": sum(r["content_length"] for r in rows),
            "validation_errors": dict(errors), "prompt_version": rubric.PROMPT_VERSION,
            "model_load_seconds_this_invocation": model_load_seconds, "generation_this_invocation": metrics,
            "document_report_sha256": _sha256(path), "requests_sha256": _sha256(request_path),
            "scope": "Complete means parsed, evidence-validated coverage only, not validated classification accuracy "
                     "or final selection. Character counts cover non-overlapping focal spans; context is not credited. "
                     "All selection decisions remain null pending pilot review."})
        LOG.info("PILOT %s: %d documents; %d unresolved spans. Reports: %s", "COMPLETE" if complete else "INCOMPLETE",
                 len(rows), len(pending), output)
        if errors:
            LOG.warning("Validation failures: %s", dict(errors))
        return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())

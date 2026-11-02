"""Read-only exhaustive final-corpus diagnostics, with reproducible review samples.

Every row and field is visited. Content hashes and token lengths are freshly
recomputed. Diagnostic text matches are review leads, never authorship claims
or keep/drop rules. The only numeric quality cutoffs are the existing approved
web policy; elsewhere the same metrics are reported without excluding records.
No corpus files, filtering decisions, or release markers are changed.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import heapq
import importlib.metadata
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit

import duckdb
import pyarrow.parquet as pq
import yaml

from ingest.connectors.logs.cloudtrail import _extract_identity_short
from ingest.utils import compute_content_hash, compute_token_count
from scripts.release.export_canonical import BASE_FIELDS
from scripts.web_filter.run import quality_features

VERSION = "exhaustive-final-inspection-v1"
# Sample sizes bound report size only. They never affect corpus selection.
SAMPLE_PER_SOURCE = 8
FLAG_SAMPLES_PER_SOURCE = 3
SNIPPET_CHARS = 9000
PATTERNS = {
    "ai_self_identification": r"\b(?:as an? (?:ai|artificial intelligence|large language) (?:language )?model|as an ai assistant|i am an ai (?:language )?model)\b",
    "ai_generation_disclosure": r"\b(?:generated|written|created) (?:by|with|using) (?:chatgpt|gpt-?[34]|an? (?:ai|large language) (?:language )?model|claude)\b",
    "assistant_refusal": r"\b(?:i(?:'m| am) sorry,? (?:but )?i (?:cannot|can't)|i (?:cannot|can't) (?:assist|help) (?:you )?with (?:that|this) request)\b",
    "chat_control_tokens": r"<\|(?:im_start|im_end|assistant|system|user|endoftext|eot_id|start_header_id)\|>|\[/?INST\]",
    "pipeline_prompt_marker": r"Evaluate this (?:arXiv|Q&A|YouTube)|qwen_should_keep|qwen_security_relevance|security_relevance_at_least_2_and_quality_at_least_2",
    "placeholder_prose": r"\blorem ipsum\b|\[(?:insert (?:text|content|title|description) here|your (?:text|content) here)\]",
    "web_error_or_challenge": r"\b(?:enable javascript and cookies to continue|checking your browser before accessing|verify you are human|access denied|page not found|404 not found)\b",
    "cookie_or_navigation": r"\b(?:accept all cookies|manage cookie preferences|skip to (?:main )?content|subscribe to our newsletter)\b",
    "unexpanded_latex": r"\\(?:documentclass|begin\{document\}|usepackage|newcommand|includegraphics|input\{|bibliography\{)",
    "html_markup": r"</?(?:html|body|script|iframe|div|p|br|span)(?:\s|>)",
    "pdf_glyph_marker": r"\(cid:\d+\)|\b(?:UnicodeDecodeError|pdftotext:|PDFSyntaxError)\b",
    "private_key_marker": r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----",
}
REGEX = {k: re.compile(v, re.I) for k, v in PATTERNS.items()}
BAD = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd]")
PLACEHOLDERS = {"null", "none", "nan", "n/a", "tbd", "todo", "unknown", "untitled", "placeholder"}
SMALL_VALUES = {
    "license", "severity", "rule_level", "rule_format", "framework", "source_format",
    "qwen_security_relevance", "qwen_quality", "qwen_should_keep", "qwen_model_should_keep",
    "qwen_parse_status", "qwen_keep_policy", "qwen_keep_policy_passed", "qwen_model",
    "qwen_prompt_version", "qwen_task", "qwen_input_kind", "kind", "dataset_revision",
}
BOOL_FIELDS = {"closed", "has_accepted_answer", "exploited_in_wild", "has_errors", "qwen_should_keep", "qwen_model_should_keep", "qwen_keep_policy_passed"}
LIST_FIELDS = {"tags", "authors", "categories", "cwe_ids", "principals", "actions", "aws_services", "regions"}
NONNEGATIVE = {"answer_count", "event_count", "session_duration_seconds", "source_row"}
STRING_FIELDS = set(BASE_FIELDS) - {"content_length", "ingested_at", "published_at", "raw"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def rank(identity):
    return (int(hashlib.sha256((VERSION + identity).encode()).hexdigest()[:16], 16) + 1) / (2**64 + 1)


def add_sample(heaps, key, priority, obj, limit):
    heap = heaps[key]
    item = (-priority, obj["file"], obj["row"], obj)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item[:3] > heap[0][:3]:
        heapq.heapreplace(heap, item)


def excerpts(text):
    if len(text) <= SNIPPET_CHARS:
        return [{"offset": 0, "text": text}]
    n = SNIPPET_CHARS // 3
    return [{"offset": i, "text": text[i:i+n]} for i in (0, len(text)//2-n//2, len(text)-n)]


def cloudtrail_errors(row, content):
    errors = []
    if "\n## Events\n" not in content:
        return ["cloudtrail_missing_events_section"]
    try:
        events = [json.loads(s) for s in content.split("\n## Events\n", 1)[1].splitlines() if s.strip()]
    except (ValueError, TypeError):
        return ["cloudtrail_invalid_event_json"]
    if any(not isinstance(e, dict) for e in events):
        return ["cloudtrail_non_object_event"]
    if len(events) != row.get("event_count"):
        errors.append("cloudtrail_event_count_mismatch")
    for col, key in (("actions", "eventName"), ("aws_services", "eventSource"), ("regions", "awsRegion")):
        actual = {e.get(key, "") for e in events if col != "regions" or e.get(key)}
        if actual != set(row.get(col) or []):
            errors.append("cloudtrail_" + col + "_mismatch")
    if {_extract_identity_short(e) for e in events} != set(row.get("principals") or []):
        errors.append("cloudtrail_principals_mismatch")
    if any("errorCode" in e for e in events) != row.get("has_errors"):
        errors.append("cloudtrail_has_errors_mismatch")
    if any(e.get("sourceIPAddress", "") != row.get("source_ip") for e in events):
        errors.append("cloudtrail_source_ip_mismatch")
    try:
        stamps = [datetime.fromisoformat(e["eventTime"].replace("Z", "+00:00")) for e in events]
        if not stamps or stamps != sorted(stamps):
            errors.append("cloudtrail_event_order_invalid")
        elif int((stamps[-1]-stamps[0]).total_seconds()) != row.get("session_duration_seconds"):
            errors.append("cloudtrail_duration_mismatch")
        if stamps and stamps[0] != row.get("published_at"):
            errors.append("cloudtrail_start_mismatch")
    except (KeyError, ValueError, TypeError):
        errors.append("cloudtrail_invalid_event_time")
    return errors


def inspect_row(row, policy, recount=True):
    errors, flags = [], []
    source = row.get("source_id", "")
    for name in ("source_id", "source_record_id", "record_id", "content", "content_hash", "license"):
        if not isinstance(row.get(name), str) or not row[name].strip():
            errors.append("invalid_" + name)
    for name in STRING_FIELDS:
        if row.get(name) is not None and not isinstance(row[name], str):
            errors.append("wrong_type_" + name)
    for name in BOOL_FIELDS:
        if row.get(name) is not None and type(row[name]) is not bool:
            errors.append("wrong_type_" + name)
    for name in LIST_FIELDS:
        if row.get(name) is not None:
            if not isinstance(row[name], list) or any(not isinstance(x, str) for x in row[name]):
                errors.append("wrong_type_" + name)
            elif any(not x.strip() for x in row[name]):
                flags.append("blank_list_element_" + name)
    for name in NONNEGATIVE:
        if row.get(name) is not None and (type(row[name]) is not int or row[name] < 0):
            errors.append("invalid_" + name)
    length = row.get("content_length")
    if type(length) is not int or length <= 0:
        errors.append("invalid_content_length")
        length = 0
    if row.get("record_id") != f"{source}:{row.get('source_record_id')}":
        errors.append("record_id_namespace_mismatch")
    for name in ("ingested_at", "published_at"):
        value = row.get(name)
        if value is None:
            if name == "ingested_at":
                errors.append("missing_ingested_at")
        elif not isinstance(value, datetime):
            errors.append("invalid_" + name)
        elif value.utcoffset() is None:
            flags.append("timezone_missing_" + name)
        elif value > datetime.now(timezone.utc):
            flags.append("future_" + name)
    if row.get("source_url"):
        try:
            url = urlsplit(row["source_url"])
            if url.scheme not in ("http", "https") or not url.hostname:
                flags.append("invalid_source_url")
            if re.search(r"\s", row["source_url"]):
                flags.append("whitespace_source_url")
        except ValueError:
            flags.append("invalid_source_url")
    for k, v in row.items():
        if isinstance(v, float) and not math.isfinite(v):
            errors.append("nonfinite_" + k)
        if isinstance(v, str) and k not in ("content", "raw", "abstract", "rule_source", "qwen_raw_response", "qwen_reason"):
            if v.strip().lower() in PLACEHOLDERS:
                flags.append("placeholder_value_" + k)
            if BAD.search(v):
                flags.append("control_or_replacement_in_" + k)
    if row.get("raw") is not None:
        try:
            raw = json.loads(row["raw"]) if isinstance(row["raw"], str) else row["raw"]
            if not isinstance(raw, dict):
                errors.append("raw_not_object")
        except (ValueError, TypeError):
            raw = {}
            errors.append("raw_invalid_json")
    else:
        raw = {}
    if source in ("nvd", "cisa-kev"):
        cve = row.get("cve_id")
        if not isinstance(cve, str) or not re.fullmatch(r"CVE-\d{4}-\d{4,}", cve):
            errors.append("invalid_cve_id")
        elif cve != row.get("source_record_id"):
            errors.append("cve_id_mismatch")
        score = row.get("cvss_score")
        if score is not None and (type(score) not in (int, float) or not 0 <= score <= 10):
            errors.append("cvss_out_of_range")
        if raw.get("cve", {}).get("vulnStatus") == "Rejected":
            flags.append("nvd_rejected_status")
    text = row.get("content") if isinstance(row.get("content"), str) else ""
    if compute_content_hash(text) != row.get("content_hash"):
        errors.append("content_hash_mismatch")
    recomputed = compute_token_count(text) if recount else None
    if recount and recomputed != length:
        errors.append("content_length_mismatch")
    if text.strip().lower() in PLACEHOLDERS | {"[deleted]", "[removed]"}:
        errors.append("content_is_placeholder")
    features = quality_features(text)
    if features["repeated_line_fraction"] >= policy["max_repeated_line_fraction"]:
        flags.append("heavy_repeated_lines")
    if features["bad_character_fraction"] >= policy["max_bad_character_fraction"]:
        flags.append("heavy_bad_characters")
    if "\x00" in text:
        flags.append("nul_in_content")
    matches = {}
    for name, pattern in REGEX.items():
        match = pattern.search(text)
        if match:
            flags.append(name)
            matches[name] = {"offset": match.start(), "excerpt": text[max(0, match.start()-180):match.end()+240]}
    if "\ufffd" in text:
        flags.append("replacement_character")
    if source == "sigma":
        try:
            rule = yaml.safe_load(row.get("rule_source") or "")
            if not isinstance(rule, dict):
                errors.append("sigma_yaml_not_object")
            else:
                for key in ("id", "title", "logsource", "detection"):
                    if not rule.get(key):
                        errors.append("sigma_missing_" + key)
                if rule.get("id") != row.get("rule_id"):
                    errors.append("sigma_rule_id_mismatch")
                if row.get("rule_source") not in text:
                    errors.append("sigma_yaml_missing_from_content")
        except yaml.YAMLError:
            errors.append("sigma_invalid_yaml")
    if source == "cloudtrail-flaws":
        errors.extend(cloudtrail_errors(row, text))
    if source == "arxiv":
        if row.get("arxiv_id") != row.get("source_record_id"):
            errors.append("arxiv_id_mismatch")
        for key in ("authors", "abstract", "categories", "source_format"):
            if not row.get(key):
                flags.append("arxiv_missing_" + key)
        if "## Abstract" in text and not re.search(r"## (?:Full (?:Text|Paper)|Paper|Body|Introduction)|\b1\s+Introduction\b", text, re.I):
            flags.append("arxiv_body_heading_not_recognized")
    if row.get("qwen_parse_status") is not None:
        if row["qwen_parse_status"] not in ("ok", "extracted_json"):
            errors.append("qwen_parse_not_accepted")
        for k in ("qwen_security_relevance", "qwen_quality"):
            if type(row.get(k)) is not int or not 0 <= row[k] <= 3:
                errors.append("invalid_" + k)
        if row.get("qwen_should_keep") is not True:
            flags.append("qwen_selection_not_true")
        if row.get("qwen_keep_policy_passed") is False:
            flags.append("qwen_keep_policy_failed")
        for k in ("qwen_model", "qwen_prompt_version", "qwen_scored_at", "qwen_reason"):
            if not row.get(k):
                flags.append("missing_" + k)
        # Metadata generated by the classifier is expected, but not in content.
        for k in ("qwen_reason", "qwen_raw_response"):
            value = row.get(k)
            if value and len(value.split()) >= 8 and value in text:
                flags.append(k + "_verbatim_in_content")
        try:
            response = json.loads(row.get("qwen_raw_response") or "")
            if isinstance(response, dict):
                for original, stored in (("security_relevance", "qwen_security_relevance"), ("quality", "qwen_quality")):
                    if original in response and str(response[original]) != str(row[stored]):
                        flags.append("qwen_response_score_mismatch")
        except (TypeError, ValueError):
            flags.append("qwen_response_not_standalone_json")
    features.update(recomputed_tokens=recomputed, flags=sorted(set(flags)), errors=sorted(set(errors)), matches=matches)
    return features


def profile_value(profile, value, categorical=False):
    profile["rows"] += 1
    if value is None:
        profile["null"] += 1
        return
    profile["non_null"] += 1
    if isinstance(value, str):
        profile["blank"] += not value.strip()
        profile["trim_differs"] += value != value.strip()
    if isinstance(value, list):
        profile["empty_list"] += not value
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        profile["min"] = min(profile.get("min", value), value)
        profile["max"] = max(profile.get("max", value), value)
    if categorical:
        key = "value:" + str(value)
        profile[key] += 1


def inspect_group(args):
    group_id, entries, root, output, policy, recount = args
    root, output = Path(root), Path(output)
    path = output / "chunks" / f"{group_id:04d}.jsonl.gz"
    counts, by_source, profiles, schemas, samples = Counter(), defaultdict(Counter), {}, {}, defaultdict(list)
    with gzip.open(path, "wt", compresslevel=1) as target:
        for ordinal, entry in entries:
            file = root / entry["relative_path"]
            if file.resolve().parent != (root / "records").resolve():
                raise ValueError("Unexpected input path")
            if sha(file) != entry["sha256"]:
                raise ValueError(f"Input checksum differs: {file}")
            file_count, file_tokens = 0, 0
            with pq.ParquetFile(file) as pf:
                if not set(BASE_FIELDS) <= set(pf.schema_arrow.names):
                    raise ValueError(f"Missing canonical columns: {file}")
                schema = [(f.name, str(f.type)) for f in pf.schema_arrow]
                for batch in pf.iter_batches(batch_size=64, use_threads=False):
                    for row in batch.to_pylist():
                        source = row["source_id"]
                        s = by_source[source]
                        features = inspect_row(row, policy, recount)
                        text, length = row["content"], row["content_length"]
                        metric = dict(file=ordinal, row=file_count, source=source, record_id=row["record_id"], content_hash=row["content_hash"], tokens=length, **{k:v for k,v in features.items() if k != "matches"})
                        target.write(json.dumps(metric, ensure_ascii=False) + "\n")
                        s.update(records=1, tokens=length, characters=len(text), structural_error_records=bool(features["errors"]), flagged_records=bool(features["flags"]))
                        s.update({"error:"+f:1 for f in features["errors"]})
                        s.update({"flag:"+f:1 for f in features["flags"]})
                        for name, value in row.items():
                            p = profiles.setdefault((source, name), Counter())
                            profile_value(p, value, name in SMALL_VALUES)
                        for name, dtype in schema:
                            schemas.setdefault(source, {}).setdefault(name, set()).add(dtype)
                        identity = str(ordinal)+":"+str(file_count)+":"+row["record_id"]
                        u = rank(identity)
                        sample = dict(metric, title=row.get("title"), source_url=row.get("source_url"), license=row.get("license"), matches=features["matches"], excerpts=excerpts(text), content_complete_in_excerpts=len(text)<=SNIPPET_CHARS)
                        for mode, priority in (("uniform", u), ("token_weighted", -math.log(u)/max(length, 1))):
                            add_sample(samples, (source, mode), priority, sample, SAMPLE_PER_SOURCE)
                        for flag in features["flags"] + features["errors"]:
                            add_sample(samples, (source, flag), u, sample, FLAG_SAMPLES_PER_SOURCE)
                        for metric_name in ("tokens", "characters", "repeated_line_fraction", "bad_character_fraction"):
                            value = length if metric_name == "tokens" else features[metric_name]
                            for label, priority in (("max", -value), ("min", value)):
                                add_sample(samples, (source, metric_name+"_"+label), priority, sample, 2)
                        file_count += 1
                        file_tokens += length
            if file_count != entry["records"] or file_tokens != entry["tokens"]:
                raise ValueError(f"Manifest counts differ: {file}")
            if sha(file) != entry["sha256"]:
                raise ValueError(f"Input changed during scan: {file}")
            counts.update(files=1, records=file_count, tokens=file_tokens)
    result = dict(counts=counts, sources=by_source, profiles=[dict(source=s, field=k, **v) for (s,k),v in profiles.items()], schemas={s: {k: sorted(v) for k,v in cols.items()} for s,cols in schemas.items()})
    save(output / "chunks" / f"{group_id:04d}.summary.json", result)
    with gzip.open(output / "chunks" / f"{group_id:04d}.samples.jsonl.gz", "wt", compresslevel=1) as f:
        for (source, category), heap in samples.items():
            for neg, _, _, obj in heap:
                f.write(json.dumps(dict(source=source, category=category, priority=-neg, sample=obj), ensure_ascii=False, default=str)+"\n")
    return dict(group=group_id, counts=counts)


def run(root, output, workers, recount=True):
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise ValueError("Use a new output directory")
    if output.is_relative_to(root):
        raise ValueError("Do not write audit outputs inside corpus")
    manifest = json.loads((root / "corpus-manifest.json").read_text())
    marker = json.loads((root / "final-corpus-complete.json").read_text())
    if sha(root / "corpus-manifest.json") != marker["manifest_sha256"]:
        raise ValueError("Completion marker does not match manifest")
    policy_path = Path(__file__).parents[1] / "web_filter/policy.json"
    policy = json.loads(policy_path.read_text())
    (output / "chunks").mkdir(parents=True)
    binding = dict(version=VERSION, started_at=datetime.now(timezone.utc).isoformat(), corpus=str(root), manifest_sha256=sha(root / "corpus-manifest.json"), code_sha256=sha(__file__), web_policy=policy, patterns=PATTERNS, uniform_samples_per_source=SAMPLE_PER_SOURCE, token_weighted_samples_per_source=SAMPLE_PER_SOURCE, diagnostic_flag_samples_per_source=FLAG_SAMPLES_PER_SOURCE, token_counts_freshly_recomputed=recount, python_packages={p:importlib.metadata.version(p) for p in ("pyarrow", "tiktoken", "duckdb", "PyYAML")}, scope="Read-only exhaustive mechanical scan plus assistant review samples. No authorship classifier, no new exclusion thresholds, no factual correctness guarantee or release approval.")
    save(output / "binding.json", binding)
    indexed = list(enumerate(manifest["files"]))
    groups = [[entry] for entry in indexed[:62]] + [indexed[i:i+128] for i in range(62, len(indexed), 128)]
    # Large files first so individual baseline files do not become late stragglers.
    groups.sort(key=lambda g: -sum(e["tokens"] for _,e in g))
    totals = Counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(inspect_group, (i, g, str(root), str(output), policy, recount)) for i,g in enumerate(groups)]
        for future in as_completed(futures):
            result = future.result()
            totals.update(result["counts"])
            save(output / "progress.json", dict(complete=False, **totals))
            print(json.dumps(dict(group=result["group"], **totals)), flush=True)
    by_source, profiles, schemas, heaps = defaultdict(Counter), {}, defaultdict(dict), defaultdict(list)
    for path in sorted((output / "chunks").glob("*.summary.json")):
        item = json.loads(path.read_text())
        for source, values in item["sources"].items():
            by_source[source].update(values)
        for p in item["profiles"]:
            source, field = p.pop("source"), p.pop("field")
            aggregate = profiles.setdefault((source,field), Counter())
            for k,v in p.items():
                if k in ("min", "max"):
                    aggregate[k] = (min if k == "min" else max)(aggregate.get(k,v),v)
                else:
                    aggregate[k] += v
        for source, fields in item["schemas"].items():
            for field, dtypes in fields.items():
                schemas[source].setdefault(field,set()).update(dtypes)
    for path in sorted((output / "chunks").glob("*.samples.jsonl.gz")):
        with gzip.open(path, "rt") as f:
            for line in f:
                x = json.loads(line)
                limit = SAMPLE_PER_SOURCE if x["category"] in ("uniform", "token_weighted") else FLAG_SAMPLES_PER_SOURCE
                add_sample(heaps, (x["source"],x["category"]), x["priority"], x["sample"], limit)
    with (output / "review-samples.jsonl").open("w") as f:
        for (source,category), heap in sorted(heaps.items()):
            for _,_,_,sample in sorted(heap, reverse=True):
                f.write(json.dumps(dict(source=source,category=category,sample=sample),ensure_ascii=False)+"\n")
    save(output / "field-profiles.json", [dict(source=s,field=f,**v) for (s,f),v in sorted(profiles.items())])
    save(output / "schemas.json", {s:{k:sorted(v) for k,v in cols.items()} for s,cols in schemas.items()})
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    files = [str(p) for p in sorted((output / "chunks").glob("*.jsonl.gz")) if not p.name.endswith(".samples.jsonl.gz")]
    relation = con.read_json(files, format="newline_delimited", union_by_name=True, sample_size=-1)
    relation.write_parquet(str(output / "record-diagnostics.parquet"), compression="zstd")
    con.read_parquet(str(output / "record-diagnostics.parquet")).create_view("diagnostics")
    dup = {name:con.execute(f"SELECT count(*) FROM (SELECT {col} FROM diagnostics GROUP BY {col} HAVING count(*)>1)").fetchone()[0] for name,col in (("record_id_groups","record_id"),("content_hash_groups","content_hash"))}
    stats = con.execute("SELECT source, count(*) records, sum(tokens) tokens, min(tokens) min_tokens, quantile_cont(tokens, [0.01,0.5,0.95,0.99]) token_quantiles, max(tokens) max_tokens, max(repeated_line_fraction) max_repeated_line_fraction FROM diagnostics GROUP BY source ORDER BY source").fetchall()
    columns = [d[0] for d in con.description]
    save(output / "source-statistics.json", [dict(zip(columns,r)) for r in stats])
    structural = sum(x["structural_error_records"] for x in by_source.values())
    summary = dict(binding, completed_at=datetime.now(timezone.utc).isoformat(), exhaustive_scan_complete=True, totals=dict(totals), duplicate_groups=dup, structural_error_records=structural, by_source=by_source, content_authorship_established=False, release_ready=False, source_selection_changed=False)
    if totals["records"] != manifest["totals"]["records"] or totals["tokens"] != manifest["totals"]["tokens"]:
        raise ValueError("Aggregate totals mismatch")
    save(output / "summary.json", summary)
    save(output / "progress.json", dict(complete=True, **totals))
    print(json.dumps({k:v for k,v in summary.items() if k in ("totals","structural_error_records","duplicate_groups")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("Use 1-8 workers")
    run(args.corpus, args.output, args.workers)


if __name__ == "__main__":
    main()

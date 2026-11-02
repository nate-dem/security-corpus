"""Restore previously generated filtering evidence; never rerun inference.

Fetched web scores must match the saved scoring audit. YouTube exports must
match worker summaries in the frozen CPU audit. Join retained records by source,
immutable input locator and content hash for web, and bound input slot plus
content hash for YouTube. Output native scores/decisions and spans separately.
"""

import argparse
from collections import Counter, defaultdict
import json
import logging
import os
from pathlib import Path
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.release import finalize_local as final
from scripts.release.correct_licenses import write
from scripts.curation_single import rubric

LOG = logging.getLogger(__name__)
SOURCES = {"primus-fineweb", "redsage-cfw", "youtube-commons"}


def evidence_plan(project):
    web = Path(os.environ.get("CYBEROPEN_WEB_AUDIT", project / "reports/curation/web-semantic-v1"))
    snapshot = Path(os.environ.get("CYBEROPEN_FINALIZED", project / "data/finalized"))
    summary = final.read(web / "review/summary.json")
    parts = final.read(web / "review/parts.json")
    final.require(summary["complete"], "Web audit incomplete")
    final.require(final.sha(web / "plan.json") == summary["plan_sha256"], "Web plan differs")
    final.require(final.sha(web / "review/parts.json") == summary["files"]["parts.json"], "Web audit differs")
    frozen = final.read(snapshot / "audit/review/frozen.json")
    youtube_audit = final.read(snapshot / "audit/review/summary.json")
    final.require(youtube_audit["audit_complete"]
                  and final.sha(snapshot / "audit/review/frozen.json") == youtube_audit["files"]["frozen.json"],
                  "Frozen YouTube audit differs")
    youtube = final.read(snapshot / "audit/review/plan.json")
    final.require(final.sha(snapshot / "audit/review/plan.json") == frozen["plan_sha256"], "YouTube plan differs")
    requests = {}
    for part in parts:
        final.require(part["complete"] and part["plan_sha256"] == summary["plan_sha256"], "Web part differs")
        if any(g["route"] == "scored" and g["score_band"] == 4 and g["records"] for g in part["groups"]):
            requests[f"web-semantic-v1/parts/{part['index']:05d}/scores.parquet"] = part["scores_sha256"]
    for part in frozen["parts"]:
        prefix = f"single-pass-v1/parts/{part['slot']:05d}"
        requests[prefix + "/summary.json"] = part["summary_sha256"]
        for name in ("decisions.parquet", "spans.parquet", "work/score/run-config.json"):
            requests[prefix + "/" + name] = None
    return dict(remote_root=str(Path(os.environ.get("CYBEROPEN_WORK_DIR", project / "data")) / "curation"), files=requests,
                web_plan_sha256=summary["plan_sha256"], model_sha256=summary["model_sha256"],
                youtube_plan_sha256=frozen["plan_sha256"], youtube_model=youtube["model"],
                frozen=frozen, snapshot=str(snapshot))


def web_key(row):
    return (row["source"], row["dataset_revision"], row["source_shard"],
            row["source_row"], row["content_hash"])


def check_web_score(score, plan):
    final.require(score["plan_sha256"] == plan["web_plan_sha256"]
                  and score["model_sha256"] == plan["model_sha256"], "Web provenance differs")
    final.require(score["route"] == "scored" and score["joint_score"] >= 0.80
                  and score["joint_score"] == min(score["domain_score"], score["substance_score"]),
                  "Retained web record fails its approved selection")
    final.require(score["source"] != "redsage-cfw" or score["upstream_relevant"] == "true",
                  "Retained RedSage record lacks selected relevance flag")


def check_youtube_spans(rows, decision, characters, plan, config_sha):
    rows = sorted(rows, key=lambda row: row["start"])
    final.require(rows and rows[0]["start"] == 0 and rows[-1]["end"] == characters,
                  "YouTube span coverage incomplete")
    end = 0
    for row in rows:
        final.require(row["content_hash"] == decision["content_hash"]
                      and row["start"] == end and row["end"] > end, "YouTube span gap/overlap")
        final.require(row["model"] == plan["youtube_model"]["model"]
                      and row["model_revision"] == plan["youtube_model"]["revision"]
                      and row["run_config_sha256"] == config_sha,
                      "YouTube span provenance differs")
        final.require(row["prompt_version"] == rubric.VERSION, "YouTube prompt version differs")
        final.require(rubric.route(row, row["parse_status"]) == "eligible",
                      "Retained YouTube span not eligible")
        end = row["end"]
    final.require(decision["route"] == "eligible" and decision["scoring_complete"]
                  and decision["plan_sha256"] == plan["youtube_plan_sha256"]
                  and decision["run_config_sha256"] == config_sha, "YouTube decision differs")


class Writer:
    def __init__(self, path):
        self.path, self.writer, self.records = path, None, 0

    def append(self, table):
        if not len(table):
            return
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        final.require(table.schema.equals(self.writer.schema), "Evidence schema changed")
        self.writer.write_table(table)
        self.records += len(table)

    def close(self):
        if self.writer:
            self.writer.close()


def selected_table(table, indices, identities):
    out = table.take(pa.array(indices, type=pa.int64()))
    for name, dtype in (("record_id", pa.string()), ("final_file", pa.int64()), ("final_row", pa.int64())):
        out = out.append_column(name, pa.array([i[name] for i in identities], type=dtype))
    return out


def package(project, corpus, fetched, output):
    project, corpus, fetched, output = [p.resolve() for p in (project, corpus, fetched, output)]
    final.require(not output.exists() and not output.is_relative_to(corpus), "Use a new external evidence directory")
    prepared, manifest = final.verify_prepared(corpus)
    plan = evidence_plan(project)
    missing = [name for name in plan["files"] if not (fetched / name).is_file()]
    final.require(not missing, f"Missing {len(missing)} evidence files; fetch them first")
    output.mkdir(parents=True)
    write(output / "summary.json", dict(complete=False))
    write(output / "input-plan.json", plan)
    db = sqlite3.connect(output / "retained-index.sqlite")
    db.execute("CREATE TABLE retained (source TEXT,revision TEXT,shard TEXT,source_row INTEGER,hash TEXT,record_id TEXT UNIQUE,file INTEGER,row INTEGER,tokens INTEGER,chars INTEGER,slot INTEGER,matched INTEGER DEFAULT 0,PRIMARY KEY(source,revision,shard,source_row,hash))")
    db.execute("CREATE INDEX youtube_lookup ON retained(slot,hash)")
    dedup = final.read(Path(plan["snapshot"]) / "dedup/plan.json")
    expected = Counter()
    for ordinal, entry in enumerate(manifest["files"]):
        path = final.contained(corpus, entry["relative_path"])
        final.require(final.sha(path) == entry["sha256"], "Corpus payload changed")
        with pq.ParquetFile(path) as pf:
            if "dataset_revision" not in pf.schema_arrow.names:
                continue
            source = pf.read(columns=["source_id"])["source_id"][0].as_py() if pf.metadata.num_rows else None
            if source not in SOURCES:
                continue
            slot = None
            if source == "youtube-commons":
                old_index = int(Path(entry["original_remote_path"]).stem)
                slot = dedup["files"][old_index]["slot"]
            offset = 0
            for batch in pf.iter_batches(batch_size=128, columns=["source_id", "dataset_revision", "source_shard", "source_row", "content_hash", "record_id", "content_length", "content"]):
                for i, row in enumerate(batch.to_pylist()):
                    db.execute("INSERT INTO retained VALUES (?,?,?,?,?,?,?,?,?,?,?,0)",
                               (row["source_id"], row["dataset_revision"], row["source_shard"], row["source_row"], row["content_hash"], row["record_id"], ordinal, offset+i, row["content_length"], len(row["content"]), slot))
                    expected[source] += 1
                offset += len(batch)
        if ordinal % 250 == 0:
            db.commit()
    db.commit()
    writers = {name: Writer(output / (name + ".parquet")) for name in ("web-records", "youtube-records", "youtube-spans")}
    verified_files = {}
    try:
        for number, (name, digest) in enumerate(plan["files"].items()):
            if not name.startswith("web-"):
                continue
            path = final.contained(fetched, name)
            final.require(final.sha(path) == digest, "Web scores differ from historical audit")
            table = pq.ParquetFile(path).read()
            indices, identities = [], []
            for index, row in enumerate(table.to_pylist()):
                match = db.execute("SELECT record_id,file,row,tokens,matched FROM retained WHERE source=? AND revision=? AND shard=? AND source_row=? AND hash=?", web_key(row)).fetchone()
                if match is None:
                    continue
                rid, file, row_number, tokens, matched = match
                final.require(not matched and tokens == row["content_length"], "Web evidence duplicated or tokens differ")
                check_web_score(row, plan)
                indices.append(index)
                identities.append(dict(record_id=rid, final_file=file, final_row=row_number))
                db.execute("UPDATE retained SET matched=1 WHERE record_id=?", (rid,))
            writers["web-records"].append(selected_table(table, indices, identities))
            verified_files[name] = digest
            if number % 250 == 0:
                db.commit()
                LOG.info("Web evidence %s files", number+1)
        for part in plan["frozen"]["parts"]:
            slot = part["slot"]
            relative = f"single-pass-v1/parts/{slot:05d}"
            folder = final.contained(fetched, relative)
            final.require(final.sha(folder / "summary.json") == part["summary_sha256"], "Frozen YouTube summary differs")
            summary = final.read(folder / "summary.json")
            final.require(summary["complete"] and summary["workflow_finished"]
                          and summary["plan_sha256"] == plan["youtube_plan_sha256"], "YouTube part incomplete")
            verified_files[relative + "/summary.json"] = part["summary_sha256"]
            for name in ("decisions.parquet", "spans.parquet"):
                digest = final.sha(folder / name)
                final.require(digest == summary["files"][name], "YouTube export differs")
                verified_files[relative + "/" + name] = digest
            config_sha = final.sha(folder / "work/score/run-config.json")
            verified_files[relative + "/work/score/run-config.json"] = config_sha
            decisions = pq.ParquetFile(folder / "decisions.parquet").read()
            spans = pq.ParquetFile(folder / "spans.parquet").read()
            grouped = defaultdict(list)
            for i, span in enumerate(spans.to_pylist()):
                grouped[span["content_hash"]].append((i, span))
            di, dn, si, sn = [], [], [], []
            for i, row in enumerate(decisions.to_pylist()):
                matches = db.execute("SELECT record_id,file,row,tokens,chars,matched FROM retained WHERE slot=? AND hash=?", (slot, row["content_hash"])).fetchall()
                if not matches:
                    continue
                final.require(len(matches) == 1, "Ambiguous YouTube content identity")
                rid, file, row_number, tokens, chars, matched = matches[0]
                final.require(not matched and row["source"] == "youtube-commons"
                              and row["content_length"] == tokens, "YouTube identity differs")
                selected = grouped[row["content_hash"]]
                check_youtube_spans([s for _, s in selected], row, chars, plan, config_sha)
                note = dict(record_id=rid, final_file=file, final_row=row_number)
                di.append(i)
                dn.append(note)
                si.extend(index for index, _ in selected)
                sn.extend([note] * len(selected))
                db.execute("UPDATE retained SET matched=1 WHERE record_id=?", (rid,))
            writers["youtube-records"].append(selected_table(decisions, di, dn))
            writers["youtube-spans"].append(selected_table(spans, si, sn))
            db.commit()
        missing = db.execute("SELECT source,count(*) FROM retained WHERE matched=0 GROUP BY source").fetchall()
        final.require(not missing, f"Unmatched retained filtering evidence: {missing}")
    finally:
        for writer in writers.values():
            writer.close()
        db.close()
    final.require(final.verify_prepared(corpus)[0] == prepared, "Corpus changed during join")
    report = dict(complete=True, corpus=str(corpus), manifest_sha256=prepared["manifest_sha256"],
                  retained_by_source=dict(expected), record_coverage=sum(expected.values()),
                  evidence_rows={name: w.records for name,w in writers.items()},
                  files={p.name: final.sha(p) for p in output.glob("*.parquet")},
                  verified_input_files=verified_files, selection_unchanged=True,
                  scope="Historical filtering evidence, not a new independent quality evaluation. Original null selection_decision values are preserved; selection is verified from the recorded routing/cutoff.")
    write(output / "summary.json", report)
    return {k:v for k,v in report.items() if k != "verified_input_files"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=Path.cwd())
    sub = p.add_subparsers(dest="command", required=True)
    request = sub.add_parser("request")
    request.add_argument("--output", type=Path, required=True)
    join = sub.add_parser("package")
    join.add_argument("--corpus", type=Path, required=True)
    join.add_argument("--fetched", type=Path, required=True)
    join.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "request":
        plan = evidence_plan(args.project.resolve())
        args.output.mkdir(parents=True, exist_ok=True)
        write(args.output / "fetch-plan.json", plan)
        (args.output / "transfer-files.txt").write_text("\n".join(plan["files"]) + "\n")
        print(json.dumps(dict(files=len(plan["files"]), output=str(args.output))))
    else:
        print(json.dumps(package(args.project, args.corpus, args.fetched, args.output), indent=2))

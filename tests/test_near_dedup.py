import json
import sqlite3
import zlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("datasketch")

from ingest.utils import compute_content_hash, compute_token_count
from scripts.near_dedup import core, run
from scripts.youtube.download import _write_json
from scripts.youtube.profile import _sha256


def words(count=600):
    return [f"identifier{i:04d}" for i in range(count)]


def test_exact_boundary_and_lexical_preservation():
    assert core.qualifies(9, 10)
    assert not core.qualifies(899, 1000)
    assert core.shingles("a == b") == core.shingles("a\n==\t b")
    assert core.shingles("a == b") != core.shingles("a != b")
    assert core.shingles("A == b") != core.shingles("a == b")
    assert core.shingles("x") == {"x"}
    with pytest.raises(ValueError, match="Empty"):
        core.shingles("\n\t ")
    a = core.shingles(" ".join(words()))
    b = core.shingles("\n".join(words()))
    assert core.signature(a) == core.signature(b)
    assert len(core.buckets(core.signature(a))) == 16
    reverse = core.shingles(" ".join(reversed(words())))
    assert not core.qualifies(*core.similarity(a, reverse))


def put_document(con, identifier, text, baseline):
    values = core.shingles(text)
    con.execute(
        "INSERT INTO docs VALUES (?,?,?,?,?,?,?,?,?)",
        (
            identifier,
            int(baseline),
            "baseline" if baseline else "web",
            compute_token_count(text),
            compute_content_hash(text),
            len(values),
            core.signature(values),
            zlib.compress(text.encode()),
            "{}",
        ),
    )


def test_no_transitive_removal_and_all_baseline_preserved(tmp_path):
    con = run.database(tmp_path, "a" * 64)
    a = words(1000)
    b, c = list(a), list(a)
    for i in range(20, 180, 20):
        b[i] = c[i] = "replacement"
    for i in range(220, 380, 20):
        c[i] = "replacement"
    texts = [" ".join(v) for v in (a, b, c)]
    sets = [core.shingles(t) for t in texts]
    assert core.qualifies(*core.similarity(sets[0], sets[1]))
    assert core.qualifies(*core.similarity(sets[1], sets[2]))
    assert not core.qualifies(*core.similarity(sets[0], sets[2]))
    for i, text in enumerate(texts):
        put_document(con, i, text, i == 0)
    # Force B to be considered before C, independently of tokenizer word length.
    con.execute("UPDATE docs SET tokens=3000 WHERE doc_id=1")
    con.execute("UPDATE docs SET tokens=2000 WHERE doc_id=2")
    con.commit()
    run.resolve(con, {"files": [{"records": 3}], "output": str(tmp_path)})
    assert con.execute(
        "SELECT doc_id,keep,rep FROM decisions ORDER BY doc_id"
    ).fetchall() == [(0, 1, 0), (1, 0, 0), (2, 1, 2)]
    con.close()


def test_crowded_bucket_checks_all_candidates(tmp_path, monkeypatch):
    con = run.database(tmp_path, "b" * 64)
    monkeypatch.setattr(core, "buckets", lambda _: [(0, b"crowded")])
    texts = [
        " ".join(f"family{i:02d}token{k:03d}" for k in range(30)) for i in range(40)
    ]
    for i, text in enumerate(texts):
        put_document(con, i, text, True)
    con.commit()
    run.resolve(con, {"files": [{"records": 40}], "output": str(tmp_path)})
    last = con.execute(
        "SELECT doc_id FROM docs ORDER BY tokens ASC,content_hash DESC,doc_id DESC LIMIT 1"
    ).fetchone()[0]
    put_document(con, 100, texts[last].replace(" ", "\n\n"), False)
    con.commit()
    run.resolve(con, {"files": [{"records": 41}], "output": str(tmp_path)})
    assert con.execute(
        "SELECT keep,rep,comparisons FROM decisions WHERE doc_id=100"
    ).fetchone() == (0, last, 40)
    assert (
        con.execute("SELECT count(*) FROM decisions WHERE keep=1").fetchone()[0] == 40
    )
    con.close()


@pytest.fixture
def corpus(tmp_path):
    project, selected, output = [
        tmp_path / s for s in ("project", "selected", "output")
    ]
    (project / "data").mkdir(parents=True)
    (selected / "records/primus-fineweb").mkdir(parents=True)
    (selected / "metadata/primus-fineweb").mkdir(parents=True)
    (selected / "review").mkdir()
    base = " ".join(words())
    baseline = [
        {
            "source_id": "baseline",
            "record_id": "baseline:1",
            "content": base,
            "content_length": compute_token_count(base),
            "content_hash": compute_content_hash(base),
            "source_url": "https://example.org/base",
            "license": "test-license",
        }
    ]
    path = project / "data/baseline.parquet"
    pq.write_table(pa.Table.from_pylist(baseline), path)
    new = " ".join(f"networking{k:04d}" for k in range(600))
    shorter = " ".join(new.split()[:-10])
    texts = [
        base,
        base.replace(" ", "\n"),
        new,
        shorter,
        new,
        "A distinctly unrelated small example.",
    ]
    records = [
        {
            "source": "primus-fineweb",
            "kind": "web",
            "dataset_revision": "a" * 40,
            "source_shard": "raw.jsonl",
            "source_row": i,
            "content_hash": compute_content_hash(t),
            "content_length": compute_token_count(t),
            "text": t,
            "source_copies": 1,
        }
        for i, t in enumerate(texts)
    ]
    metadata = [
        {
            k: r[k]
            for k in (
                "source",
                "dataset_revision",
                "source_shard",
                "source_row",
                "content_hash",
                "content_length",
            )
        }
        | {"url": f"https://example.org/{i}", "upstream_license": "test-license"}
        for i, r in enumerate(records)
    ]
    rp, mp = (
        selected / "records/primus-fineweb/0.parquet",
        selected / "metadata/primus-fineweb/0.parquet",
    )
    pq.write_table(pa.Table.from_pylist(records), rp)
    pq.write_table(pa.Table.from_pylist(metadata), mp)
    binding = {"plan_sha256": "c" * 64, "min_domain_and_substance_score": 0.8}
    _write_json(selected / "selection.json", binding)
    summary = {
        "complete": True,
        **binding,
        "by_source": {
            "primus-fineweb": {
                "records": len(records),
                "tokens": sum(r["content_length"] for r in records),
            }
        },
        "files": [
            {
                "path": str(rp.relative_to(selected)),
                "metadata_path": str(mp.relative_to(selected)),
                "sha256": _sha256(rp),
                "metadata_sha256": _sha256(mp),
                "source": "primus-fineweb",
                "index": 0,
                "records": len(records),
                "tokens": sum(r["content_length"] for r in records),
            }
        ],
    }
    _write_json(selected / "summary.json", summary)
    _write_json(selected / "review/summary.json", summary)
    refs = {
        "baseline_records": 1,
        "baseline_tokens": baseline[0]["content_length"],
        "baseline_files": [
            {
                "project_relative_path": "data/baseline.parquet",
                "sha256": _sha256(path),
                "records": 1,
                "recomputed_tokens": baseline[0]["content_length"],
            }
        ],
        "selected_summary_sha256": _sha256(selected / "summary.json"),
        "selected_config_sha256": _sha256(selected / "selection.json"),
    }
    return project, selected, output, refs, records, metadata


def execute(corpus):
    project, selected, output, refs, _, _ = corpus
    return run.execute(project, selected, output, workers=2, references=refs)


def test_workflow_export_lineage_samples_and_resume(corpus):
    project, selected, output, _, records, metadata = corpus
    before = {
        p: _sha256(p) for root in (project, selected) for p in root.rglob("*.parquet")
    }
    report = execute(corpus)
    assert report["complete"] and not report["release_ready"]
    assert report["baseline"]["records"] == 1 and report["baseline"]["unchanged"]
    routes = report["by_source"]["primus-fineweb"]
    assert routes["retained"]["records"] == 2
    assert routes["exact"]["records"] == 2
    assert routes["near"]["records"] == 2
    exported = (
        pq.ParquetFile(next((output / "records").rglob("*.parquet"))).read().to_pylist()
    )
    kept_metadata = (
        pq.ParquetFile(next((output / "metadata").rglob("*.parquet")))
        .read()
        .to_pylist()
    )
    assert exported == [records[2], records[5]]
    assert kept_metadata == [metadata[2], metadata[5]]
    decisions = (
        pq.ParquetFile(next((output / "decisions").glob("*.parquet")))
        .read()
        .to_pylist()
    )
    assert decisions[1]["representative_source"] == "baseline"
    assert decisions[3]["representative_hash"] == records[2]["content_hash"]
    packet = json.loads((output / "review/sample.json").read_text())
    assert len(packet["token_draws"]) == 64 and len(packet["removed_pairs"]) == 4
    assert all(p["jaccard"] >= 0.9 for p in packet["removed_pairs"])
    assert report == execute(corpus)
    assert before == {p: _sha256(p) for p in before}


def test_resume_after_decisions_committed_before_export(corpus, monkeypatch):
    original = run.finish
    monkeypatch.setattr(
        run,
        "finish",
        lambda *args: (_ for _ in ()).throw(InterruptedError("simulated interruption")),
    )
    with pytest.raises(InterruptedError):
        execute(corpus)
    output = corpus[2]
    assert not (output / "summary.json").exists()
    with sqlite3.connect(output / "state.sqlite") as con:
        assert con.execute("SELECT count(*) FROM decisions").fetchone()[0] == 7
    monkeypatch.setattr(run, "finish", original)
    assert execute(corpus)["retained_additions"]["records"] == 2


def test_input_and_output_corruption_rejected(corpus):
    execute(corpus)
    output = corpus[2]
    exported = next((output / "records").rglob("*.parquet"))
    original = exported.read_bytes()
    exported.write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="Export checkpoint"):
        execute(corpus)
    exported.write_bytes(original)
    input_file = corpus[0] / "data/baseline.parquet"
    input_file.write_bytes(input_file.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Input checksum"):
        execute(corpus)


def test_changed_signature_checkpoint_rejected(corpus):
    execute(corpus)
    path = next((corpus[2] / "signatures").rglob("*.parquet"))
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Signature checkpoint"):
        execute(corpus)


def test_reject_output_inside_inputs(corpus):
    project, selected, _, refs, _, _ = corpus
    with pytest.raises(ValueError, match="separate"):
        run.execute(project, selected, selected / "bad", references=refs)


def test_uncommitted_decisions_and_indexes_roll_back_together(tmp_path, monkeypatch):
    con = run.database(tmp_path, "d" * 64)
    for i in range(160):
        put_document(con, i, f"document{i}", True)
    con.commit()
    original = run.choose
    calls = 0

    def interrupted(*args):
        nonlocal calls
        calls += 1
        if calls == 102:
            raise InterruptedError("mid-transaction")
        return original(*args)

    monkeypatch.setattr(run, "choose", interrupted)
    plan = {"files": [{"records": 160}], "output": str(tmp_path)}
    with pytest.raises(InterruptedError):
        run.resolve(con, plan)
    con.close()
    con = run.database(tmp_path, "d" * 64)
    assert con.execute("SELECT count(*) FROM decisions").fetchone()[0] == 100
    assert con.execute("SELECT count(*) FROM bands").fetchone()[0] == 1600
    monkeypatch.setattr(run, "choose", original)
    run.resolve(con, plan)
    assert con.execute("SELECT count(*) FROM decisions").fetchone()[0] == 160
    assert con.execute("SELECT count(*) FROM bands").fetchone()[0] == 2560
    con.close()


def test_untracked_retained_file_rejected(corpus):
    execute(corpus)
    folder = corpus[2] / "records/primus-fineweb"
    (folder / "unexpected.parquet").write_bytes(
        next(folder.glob("*.parquet")).read_bytes()
    )
    with pytest.raises(ValueError, match="Unexpected"):
        execute(corpus)

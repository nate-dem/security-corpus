import io
import json
import shutil
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from classify.io import publish_outputs
from classify.qwen import parse_qwen_response
from ingest.connectors.arxiv.connector import ArxivConnector
from ingest.connectors.arxiv.latex_processing import merge_project
from scripts.arxiv.normalize_sources import _extract_source
from scripts.arxiv.select_accepted_citations import main as select_citations
from scripts.classify.merge_qwen_sidecars import main as merge
from scripts.classify.score_qwen_vllm import _load_existing_output_keys, _build_parser, _write_run_config
from classify.qwen import QwenTask
from scripts.release.audit_qwen_coverage import main as coverage


REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def decision(**updates):
    return dict({
        "source_id": "arxiv", "record_id": "arxiv:2401.00001", "content_hash": "abc",
        "qwen_should_keep": True, "qwen_parse_status": "ok",
        "qwen_model": "Qwen/Qwen3-8B", "qwen_model_revision": REVISION,
        "qwen_prompt_version": "qwen-arxiv-abstract-v1", "qwen_task": "arxiv_abstract",
        "qwen_scored_at": "2026-09-09T00:00:00+00:00", "qwen_shard_id": "0",
    }, **updates)


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


@pytest.mark.parametrize("field,value", [
    ("qwen_parse_status", None), ("qwen_model", None), ("qwen_model_revision", None),
    ("qwen_scored_at", "invalid"), ("qwen_prompt_version", ""), ("qwen_task", None),
])
def test_release_gates_reject_invalid_provenance(tmp_path, field, value):
    corpus, decisions = tmp_path / "corpus.parquet", tmp_path / "decisions.parquet"
    write(corpus, [{"source_id": "arxiv", "record_id": "arxiv:2401.00001",
                    "content_hash": "abc", "arxiv_id": "2401.00001"}])
    write(decisions, [decision(**{field: value})])
    assert coverage(["--corpus", str(corpus), "--decisions", str(decisions)]) == 2
    with pytest.raises(ValueError, match="incomplete"):
        select_citations(["--universe", str(corpus), "--decisions", str(decisions),
                          "--output", str(tmp_path / "accepted.txt")])


def test_coverage_rejects_mixed_prompts(tmp_path):
    corpus, decisions = tmp_path / "corpus.parquet", tmp_path / "decisions.parquet"
    rows = [decision(), decision(record_id="arxiv:2401.00002", qwen_prompt_version="other")]
    write(corpus, rows)
    write(decisions, rows)
    assert coverage(["--corpus", str(corpus), "--decisions", str(decisions)]) == 2


def test_resume_retries_only_unsuccessful_decisions(tmp_path):
    write(tmp_path / "part.parquet", [
        decision(),
        decision(record_id="failed", qwen_parse_status="parse_failure", qwen_should_keep=None),
        decision(record_id="undecided", qwen_should_keep=None),
    ])
    assert _load_existing_output_keys(tmp_path) == {("arxiv", "arxiv:2401.00001", "abc")}


def test_resume_refuses_unprovenanced_parts_and_changed_settings(tmp_path):
    input_path = tmp_path / "input.parquet"
    input_path.write_bytes(b"input fingerprint fixture")
    output = tmp_path / "shards"
    output.mkdir()
    args = _build_parser().parse_args(["--input", str(input_path),
                                     "--task", "qa", "--output-dir", str(output)])
    part = output / "part.parquet"
    part.touch()
    with pytest.raises(ValueError, match="no run-config"):
        _write_run_config(args, QwenTask.QA, "test")
    part.unlink()
    _write_run_config(args, QwenTask.QA, "test")
    args.seed += 1
    with pytest.raises(ValueError, match="does not match"):
        _write_run_config(args, QwenTask.QA, "test")
    args.seed -= 1
    input_path.write_bytes(b"changed prompt metadata")
    with pytest.raises(ValueError, match="does not match"):
        _write_run_config(args, QwenTask.QA, "test")


def test_merge_preserves_success_over_later_failure(tmp_path):
    shard = tmp_path / "shards" / "shard-0"
    write(shard / "part-0.parquet", [decision()])
    write(shard / "part-1.parquet", [decision(qwen_should_keep=None, qwen_parse_status="parse_failure",
                                            qwen_scored_at="2026-09-10T00:00:00+00:00")])
    (shard / "run-config.json").write_text(json.dumps({"inference": {"seed": 0}, "runtime": {"python": "3.11"}}))
    output = tmp_path / "merged.parquet"
    assert merge(["--input-dir", str(shard.parent), "--output", str(output)]) == 0
    assert pq.read_table(output)["qwen_should_keep"].to_pylist() == [True]
    write(shard / "part-2.parquet", [decision(qwen_should_keep=False)])
    before = output.read_bytes()
    with pytest.raises(ValueError, match="Conflicting"):
        merge(["--input-dir", str(shard.parent), "--output", str(output), "--overwrite"])
    assert output.read_bytes() == before


def test_latex_includes_ignore_comments_and_code_and_allow_repetition(tmp_path):
    project = tmp_path / "source"
    project.mkdir()
    (project / "main.tex").write_text(
        "\\documentclass{article}\n\\begin{document}\n"
        "% \\input{comment}\n"
        "\\begin{verbatim}\n\\input{example}\nvalue % literal\n\\end{verbatim}\n"
        "\\verb|\\input{inline}|\n"
        "\\input{body}\n\\input{body}\n\\end{document}\n"
    )
    for name in ("comment", "example", "inline"):
        (project / f"{name}.tex").write_text("MUST NOT APPEAR")
    (project / "body.tex").write_text("REAL BODY\n")
    output = tmp_path / "merged.tex"
    diagnostics = merge_project(project, output)
    text = output.read_text()
    assert "MUST NOT APPEAR" not in text
    assert r"\input{example}" in text
    assert r"\verb|\input{inline}|" in text
    assert "value % literal" in text
    assert text.count("REAL BODY") == 2
    assert diagnostics.includes_inlined == 2
    assert diagnostics.circular_includes == []


def test_rejected_compressed_tar_cannot_fall_through_to_gzip(tmp_path):
    archive = tmp_path / "paper.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        content = b"\\documentclass{article}"
        member = tarfile.TarInfo("../escape.tex")
        member.size = len(content)
        handle.addfile(member, io.BytesIO(content))
    assert _extract_source(archive, tmp_path / "extract") is False
    assert not list((tmp_path / "extract").glob("*.tex"))


def test_arxiv_requires_status_and_uses_status_selected_format(tmp_path):
    fixtures = Path(__file__).parent / "fixtures" / "arxiv"
    shutil.copytree(fixtures, tmp_path / "raw")
    paper = tmp_path / "raw/source/normalized/2401/2401.00001"
    (paper / "status.json").unlink()
    records = list(ArxivConnector().iter_records(tmp_path / "raw"))
    assert "2401.00001" not in {row["arxiv_id"] for row in records}
    (paper / "status.json").write_text(json.dumps({"completed": True, "normalizer_version": "pdf-text-v1"}))
    (paper / "main.txt").write_text("Current PDF text")
    records = list(ArxivConnector().iter_records(tmp_path / "raw"))
    row = next(row for row in records if row["arxiv_id"] == "2401.00001")
    assert row["content"] == "Current PDF text"
    assert row["source_format"] == "pdf"


def test_checkpoint_publication_rolls_back_on_failure(tmp_path):
    old = tmp_path / "old"
    staged = tmp_path / "staged"
    old.write_text("checkpoint")
    staged.write_text("replacement")
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        publish_outputs({staged: old, missing: tmp_path / "other"})
    assert old.read_text() == "checkpoint"
    assert staged.read_text() == "replacement"
    assert not list(tmp_path.glob("*.backup"))


@pytest.mark.parametrize("score", [2.9, float("inf"), [], True])
def test_parser_does_not_coerce_invalid_scores(score):
    parsed = parse_qwen_response(json.dumps({"security_relevance": score, "quality": 3,
                                            "should_keep": True, "reason": "reason"}))
    assert parsed.parse_status == "parse_failure"
    assert parsed.should_keep is None

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ingest.utils import compute_content_hash, compute_token_count
from scripts.curation import prepare
from scripts.web_corpora.profile import FEATURE_SCHEMA
from scripts.web_filter import run
from scripts.youtube.download import _write_json
from scripts.youtube.profile import _sha256


@pytest.fixture
def web_env(tmp_path):
    prepared = tmp_path / "prepared"
    project = tmp_path / "project"
    web = tmp_path / "web"
    output = tmp_path / "output"
    (project / "data").mkdir(parents=True)
    (project / "config").mkdir()
    baseline_text = "A baseline document about network socket diagnostics."
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "content": baseline_text,
                    "content_hash": compute_content_hash(baseline_text),
                }
            ]
        ),
        project / "data/baseline.parquet",
    )
    _write_json(
        project / "config/retained_baseline.json",
        {"inputs": ["data/baseline.parquet"], "expected_records": 1},
    )
    parts, inputs, summaries = [], [], {}
    for source in run.SOURCES:
        texts = [
            baseline_text,
            "cookie navigation\n" * 6,
            "Debug network sockets using system call tracing and packet captures.",
            "Damaged text " + "\ufffd" * 30,
        ]
        rows = [
            {
                "source": source,
                "kind": "web",
                "dataset_revision": "a" * 40,
                "source_shard": "data/raw.parquet",
                "source_row": i,
                "content_hash": compute_content_hash(text),
                "content_length": compute_token_count(text),
                "source_copies": 1,
                "text": text,
            }
            for i, text in enumerate(texts)
        ]
        part = prepared / f"shards/{source}/00000/00000.parquet"
        part.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=prepare.SCHEMA), part)
        parts.append(
            {
                "path": str(part.relative_to(prepared)),
                "records": len(rows),
                "tokens": sum(r["content_length"] for r in rows),
                "sha256": _sha256(part),
            }
        )
        profile = (
            web
            / source
            / ("profile-v2" if source == "primus-fineweb" else "profile-v1")
        )
        metadata = profile / "metadata/00000.parquet"
        metadata.parent.mkdir(parents=True)
        features = [
            {
                **{
                    k: r[k]
                    for k in (
                        "source",
                        "source_shard",
                        "source_row",
                        "content_hash",
                        "content_length",
                    )
                },
                "upstream_relevant": None,
                "language": "en",
                "url": "https://example.org/tutorial",
                "domain": "example.org",
                "upstream_license": "source-license",
                "text_state": "valid",
            }
            for r in rows
        ]
        pq.write_table(pa.Table.from_pylist(features, schema=FEATURE_SCHEMA), metadata)
        _write_json(
            profile / "summary.json",
            {
                "complete": True,
                "shards": [
                    {
                        "source_shard": "data/raw.parquet",
                        "metadata_sha256": _sha256(metadata),
                    }
                ],
            },
        )
        summaries[source] = _sha256(profile / "summary.json")
        inputs.append(
            {
                "source": source,
                "index": 0,
                "dataset_revision": "a" * 40,
                "source_shard": "data/raw.parquet",
                "metadata": str(metadata),
            }
        )
    (prepared / "lineage.parquet").write_bytes(b"immutable aliases")
    _write_json(
        prepared / "run-config.json", {"inputs": inputs, "source_summaries": summaries}
    )
    _write_json(
        prepared / "summary.json",
        {
            "version": "curation-inputs-v1",
            "complete": True,
            "run_config_sha256": _sha256(prepared / "run-config.json"),
            "parts": parts,
            "records": sum(p["records"] for p in parts),
            "tokens": sum(p["tokens"] for p in parts),
            "index": {"lineage_sha256": _sha256(prepared / "lineage.parquet")},
        },
    )
    plan = run.make_plan(project, prepared, web, output, run.HERE / "policy.json")
    return output / "plan.json", plan


def test_approved_damage_thresholds_do_not_become_scope_filters():
    policy = json.loads((run.HERE / "policy.json").read_text())
    run.validate_policy(policy)
    assert (
        run.route(run.quality_features("same\nsame"), policy, False)[0] == "quarantine"
    )
    assert (
        run.route(run.quality_features("\ufffd" + "x" * 19), policy, False)[0]
        == "quarantine"
    )
    assert run.route(
        run.quality_features("hello\tworld\nvalid\rline"), policy, False
    ) == ("candidate", [])
    assert run.route(run.quality_features("x"), policy, False) == ("candidate", [])
    assert run.stratum("redsage-cfw", None) == "upstream_unlabeled"
    assert run.stratum("redsage-cfw", "false") == "upstream_relevant_false"
    metrics = {
        **policy,
        "mode": "metrics_only",
        "max_repeated_line_fraction": None,
        "max_bad_character_fraction": None,
    }
    run.validate_policy(metrics)
    assert run.route(run.quality_features("\ufffd" * 10), metrics, False) == (
        "candidate",
        [],
    )
    with pytest.raises(ValueError, match="cannot silently"):
        run.validate_policy({**policy, "mode": "metrics_only"})


def test_cpu_filter_full_workflow_and_resume_preserve_inputs(web_env):
    path, plan = web_env
    prepared = Path(plan["prepared"])
    before = {p: _sha256(p) for p in prepared.rglob("*") if p.is_file()}
    index = run.baseline_index(path)
    assert index["records"] == 1 and run.baseline_index(path) == index
    for source in run.SOURCES:
        report = run.process_source(path, source, workers=1)
        assert report["complete"] and not report["release_ready"]
        assert (
            not report["inference_performed"]
            and not report["semantic_quality_validated"]
        )
        assert report["records"] == 4 and report["candidate_records"] == 1
        root = path.parent / source
        candidates = pq.read_table(
            root / "parts/00000/00000/candidates.parquet"
        ).to_pylist()
        original = pq.read_table(
            prepared / f"shards/{source}/00000/00000.parquet"
        ).to_pylist()[2]
        assert candidates == [original]
        features = pq.read_table(
            root / "parts/00000/00000/features.parquet"
        ).to_pylist()
        assert [r["route"] for r in features] == [
            "baseline_duplicate",
            "quarantine",
            "candidate",
            "quarantine",
        ]
        assert features[2]["upstream_relevant"] is None
        assert features[2]["upstream_license"] == "source-license"
        draws = json.loads((root / "review/token-draws.json").read_text())["draws"]
        cases = {
            c["case_id"]: c
            for c in json.loads((root / "review/review-packet.json").read_text())[
                "cases"
            ]
        }
        assert len(draws) == 64 * 3
        assert all(
            0 <= d["token_offset"] < cases[d["case_id"]]["content_length"]
            for d in draws
        )
        kept = [d for d in draws if d["route"] == "candidate"]
        assert (
            len({d["case_id"] for d in kept}) == 1
        )  # With-replacement draws stay repeated.
        assert run.process_source(path, source, workers=1) == report
    assert before == {p: _sha256(p) for p in prepared.rglob("*") if p.is_file()}


def test_metadata_and_baseline_corruption_refuse_resume(web_env):
    path, plan = web_env
    run.baseline_index(path)
    baseline = Path(plan["baseline_files"][0])
    original = baseline.read_bytes()
    baseline.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        run.baseline_index(path)
    baseline.write_bytes(original)
    task = plan["tasks"][0]
    Path(task["metadata"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="metadata checksum"):
        run.process_group(
            task, Path(plan["prepared"]), path.parent, plan["policy"], _sha256(path)
        )

import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]


def test_score_qwen_vllm_dry_run_writes_prompt_jsonl(tmp_path):
    input_path = tmp_path / "qa.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_id": "stackexchange-infosec",
                    "record_id": "stackexchange-infosec:1",
                    "content_hash": "h1",
                    "content": "How should I store salted password hashes?",
                    "title": "Password storage",
                    "score": 5,
                    "answer_count": 2,
                    "has_accepted_answer": True,
                    "closed": False,
                    "tags": ["passwords", "hashing"],
                }
            ]
        ),
        input_path,
    )
    prompts_path = tmp_path / "prompts.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/classify/score_qwen_vllm.py",
            "--input",
            str(input_path),
            "--task",
            "qa",
            "--dry-run",
            "--dry-run-prompts",
            str(prompts_path),
            "--max-records",
            "1",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "vLLM was not imported" in result.stdout
    row = json.loads(prompts_path.read_text(encoding="utf-8").strip())
    assert row["record_id"] == "stackexchange-infosec:1"
    assert "Return exactly one compact JSON object" in row["prompt"]


def test_score_qwen_vllm_dry_run_fails_on_missing_required_column(tmp_path):
    input_path = tmp_path / "qa_missing_content.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_id": "stackexchange-infosec",
                    "record_id": "stackexchange-infosec:1",
                    "content_hash": "h1",
                    "title": "Password storage",
                }
            ]
        ),
        input_path,
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/classify/score_qwen_vllm.py",
            "--input",
            str(input_path),
            "--task",
            "qa",
            "--dry-run",
            "--max-records",
            "1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "missing required qa prompt columns: content" in result.stderr


def test_score_qwen_vllm_dry_run_fails_on_blank_required_value(tmp_path):
    input_path = tmp_path / "qa_blank_content.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_id": "stackexchange-infosec",
                    "record_id": "stackexchange-infosec:1",
                    "content_hash": "h1",
                    "content": " ",
                }
            ]
        ),
        input_path,
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/classify/score_qwen_vllm.py",
            "--input",
            str(input_path),
            "--task",
            "qa",
            "--dry-run",
            "--max-records",
            "1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "missing/blank required qa prompt fields: content" in result.stderr


def test_score_qwen_vllm_arxiv_abstract_dry_run_does_not_require_content(tmp_path):
    input_path = tmp_path / "arxiv_abstract.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_id": "arxiv",
                    "record_id": "arxiv:2401.00001",
                    "content_hash": "h1",
                    "title": "Security Protocols",
                    "abstract": "We analyze authentication protocols.",
                    "categories": ["cs.CR"],
                }
            ]
        ),
        input_path,
    )
    prompts_path = tmp_path / "arxiv_prompts.jsonl"

    subprocess.run(
        [
            sys.executable,
            "scripts/classify/score_qwen_vllm.py",
            "--input",
            str(input_path),
            "--task",
            "arxiv-abstract",
            "--dry-run",
            "--dry-run-prompts",
            str(prompts_path),
            "--max-records",
            "1",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    row = json.loads(prompts_path.read_text(encoding="utf-8").strip())
    assert row["record_id"] == "arxiv:2401.00001"
    assert "We analyze authentication protocols" in row["prompt"]


def test_score_qwen_vllm_arxiv_source_filter_skips_non_target_missing_fields(tmp_path):
    input_path = tmp_path / "mixed_arxiv.parquet"
    schema = pa.schema(
        [
            ("source_id", pa.string()),
            ("record_id", pa.string()),
            ("content_hash", pa.string()),
            ("title", pa.string()),
            ("abstract", pa.string()),
            ("content", pa.string()),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_id": "nvd",
                    "record_id": "nvd:CVE-2026-0001",
                    "content_hash": "hnvd",
                    "content": "NVD content should not be scored.",
                },
                {
                    "source_id": "arxiv",
                    "record_id": "arxiv:2401.00001",
                    "content_hash": "harxiv",
                    "title": "Security Protocols",
                    "abstract": "We analyze authentication protocols.",
                    "content": "FULL PAPER TEXT SHOULD NOT BE LOADED FOR ABSTRACT.",
                },
            ],
            schema=schema,
        ),
        input_path,
    )
    prompts_path = tmp_path / "mixed_arxiv_prompts.jsonl"

    subprocess.run(
        [
            sys.executable,
            "scripts/classify/score_qwen_vllm.py",
            "--input",
            str(input_path),
            "--task",
            "arxiv-abstract",
            "--source-id",
            "arxiv",
            "--dry-run",
            "--dry-run-prompts",
            str(prompts_path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    rows = [
        json.loads(line)
        for line in prompts_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["record_id"] for row in rows] == ["arxiv:2401.00001"]
    assert "NVD content" not in rows[0]["prompt"]
    assert "FULL PAPER TEXT" not in rows[0]["prompt"]


def test_score_qwen_vllm_arxiv_source_filter_uses_hive_partition_before_schema_validation(tmp_path):
    root = tmp_path / "normalized"
    nvd_dir = root / "source_id=nvd"
    arxiv_dir = root / "source_id=arxiv"
    nvd_dir.mkdir(parents=True)
    arxiv_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_id": "nvd",
                    "record_id": "nvd:CVE-2026-0001",
                    "content_hash": "hnvd",
                    "content": "NVD row lacks arXiv title and abstract.",
                }
            ]
        ),
        nvd_dir / "part.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_id": "arxiv",
                    "record_id": "arxiv:2401.00001",
                    "content_hash": "harxiv",
                    "title": "Security Protocols",
                    "abstract": "We analyze authentication protocols.",
                    "content": "FULL PAPER TEXT SHOULD NOT BE IN ABSTRACT PROMPT.",
                }
            ]
        ),
        arxiv_dir / "part.parquet",
    )
    prompts_path = tmp_path / "hive_arxiv_prompts.jsonl"

    subprocess.run(
        [
            sys.executable,
            "scripts/classify/score_qwen_vllm.py",
            "--input",
            str(root),
            "--task",
            "arxiv-abstract",
            "--source-id",
            "arxiv",
            "--dry-run",
            "--dry-run-prompts",
            str(prompts_path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    rows = [
        json.loads(line)
        for line in prompts_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["record_id"] for row in rows] == ["arxiv:2401.00001"]
    assert "We analyze authentication protocols" in rows[0]["prompt"]
    assert "FULL PAPER TEXT" not in rows[0]["prompt"]


def test_score_qwen_vllm_qa_source_filter_skips_non_qa_missing_fields(tmp_path):
    input_path = tmp_path / "mixed_qa.parquet"
    schema = pa.schema(
        [
            ("source_id", pa.string()),
            ("record_id", pa.string()),
            ("content_hash", pa.string()),
            ("content", pa.string()),
            ("title", pa.string()),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_id": "nvd",
                    "record_id": "nvd:CVE-2026-0001",
                    "content_hash": "hnvd",
                },
                {
                    "source_id": "reddit-netsec",
                    "record_id": "reddit-netsec:abc",
                    "content_hash": "hreddit",
                    "content": "Detailed thread about incident response triage.",
                    "title": "Incident response triage",
                },
            ],
            schema=schema,
        ),
        input_path,
    )
    prompts_path = tmp_path / "mixed_qa_prompts.jsonl"

    subprocess.run(
        [
            sys.executable,
            "scripts/classify/score_qwen_vllm.py",
            "--input",
            str(input_path),
            "--task",
            "qa",
            "--qa-sources",
            "--dry-run",
            "--dry-run-prompts",
            str(prompts_path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    rows = [
        json.loads(line)
        for line in prompts_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["record_id"] for row in rows] == ["reddit-netsec:abc"]
    assert "incident response triage" in rows[0]["prompt"].lower()


def test_score_qwen_vllm_qa_source_filter_uses_hive_partition_before_schema_validation(tmp_path):
    root = tmp_path / "normalized"
    nvd_dir = root / "source_id=nvd"
    qa_dir = root / "source_id=stackexchange-infosec"
    nvd_dir.mkdir(parents=True)
    qa_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_id": "nvd",
                    "record_id": "nvd:CVE-2026-0001",
                    "content_hash": "hnvd",
                }
            ]
        ),
        nvd_dir / "part.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_id": "stackexchange-infosec",
                    "record_id": "stackexchange-infosec:1",
                    "content_hash": "hqa",
                    "content": "How should salted password hashes be stored?",
                    "title": "Password hashing",
                }
            ]
        ),
        qa_dir / "part.parquet",
    )
    prompts_path = tmp_path / "hive_qa_prompts.jsonl"

    subprocess.run(
        [
            sys.executable,
            "scripts/classify/score_qwen_vllm.py",
            "--input",
            str(root),
            "--task",
            "qa",
            "--qa-sources",
            "--dry-run",
            "--dry-run-prompts",
            str(prompts_path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    rows = [
        json.loads(line)
        for line in prompts_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["record_id"] for row in rows] == ["stackexchange-infosec:1"]
    assert "salted password hashes" in rows[0]["prompt"]


def test_score_qwen_vllm_source_filter_fails_when_no_matching_partition(tmp_path):
    root = tmp_path / "normalized"
    nvd_dir = root / "source_id=nvd"
    nvd_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_id": "nvd",
                    "record_id": "nvd:CVE-2026-0001",
                    "content_hash": "hnvd",
                    "content": "NVD content",
                }
            ]
        ),
        nvd_dir / "part.parquet",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/classify/score_qwen_vllm.py",
            "--input",
            str(root),
            "--task",
            "arxiv-abstract",
            "--source-id",
            "arxiv",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "No matching source_id partitions found" in result.stderr

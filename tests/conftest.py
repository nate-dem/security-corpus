from __future__ import annotations

from pathlib import Path

import pytest


def build_securityclip_fixture_index(tmp_path: Path) -> Path:
    """Build a tiny real Security Scope index (one /vulns doc, one /papers doc)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from securityclip.config import SourceSpec
    from securityclip.indexer import build_index

    parquet_path = tmp_path / "data" / "docs.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "source_id": "nvd",
            "source_record_id": "CVE-2021-44228",
            "record_id": "nvd:CVE-2021-44228",
            "title": "nvd:CVE-2021-44228",
            "content": "Apache Log4j JNDI injection vulnerability allows remote code execution.",
            "content_length": 12,
            "content_hash": "hash-nvd",
            "license": "Public Domain",
            "source_url": "https://nvd.nist.gov/vuln/detail/CVE-2021-44228",
            "cve_id": "CVE-2021-44228",
        },
        {
            "source_id": "arxiv",
            "source_record_id": "2401.00001",
            "record_id": "arxiv:2401.00001",
            "title": "Prompt Injection Study",
            "content": "# Prompt Injection Study\n\nWe evaluate prompt injection and adversarial retrieval.",
            "content_length": 18,
            "content_hash": "hash-paper",
            "license": "CC-BY-4.0",
            "arxiv_id": "2401.00001",
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)
    index_dir = tmp_path / "index"
    build_index(index_dir, source_specs=[SourceSpec("fixture", str(parquet_path))], fts_max_chars=0)
    return index_dir


@pytest.fixture
def securityclip_fixture_index(tmp_path: Path) -> Path:
    return build_securityclip_fixture_index(tmp_path)

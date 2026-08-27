from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.release.audit_source_licenses import main


def _write_rows(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_license_audit_passes_conditional_source(tmp_path):
    source = tmp_path / "nvd.parquet"
    _write_rows(
        source,
        [{
            "source_id": "nvd",
            "license": "CVE Terms of Use / NIST public data",
            "content_length": 12,
        }],
    )
    assert main([str(source)]) == 0


def test_license_audit_blocks_unlicensed_source(tmp_path):
    source = tmp_path / "reddit.parquet"
    _write_rows(
        source,
        [{
            "source_id": "reddit-netsec",
            "license": "NOASSERTION",
            "content_length": 12,
        }],
    )
    assert main([str(source)]) == 2

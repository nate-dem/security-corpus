import re
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow.compute as pc
import pytest


NVD_DIR = Path("data/nvd/normalized/source_id=nvd")
ATTACK_DIR = Path("data/mitre-attack/normalized/source_id=mitre-attack")


def _skip_if_missing(path: Path):
    if not any(path.glob("*.parquet")):
        pytest.skip(f"No Parquet files in {path}; run bulk ingest first")


@pytest.mark.data_quality
class TestSchemaConsistency:
    def test_all_nvd_files_have_same_schema(self):
        _skip_if_missing(NVD_DIR)
        schemas = [pq.ParquetFile(f).schema_arrow for f in sorted(NVD_DIR.glob("*.parquet"))]
        for s in schemas[1:]:
            assert s.equals(schemas[0], check_metadata=False)
    
    def test_nvd_and_attack_have_same_schema(self):
        _skip_if_missing(NVD_DIR)
        _skip_if_missing(ATTACK_DIR)
        nvd = next(NVD_DIR.glob("*.parquet"))
        attack = next(ATTACK_DIR.glob("*.parquet"))
        nvd_schema = pq.ParquetFile(nvd).schema_arrow
        attack_schema = pq.ParquetFile(attack).schema_arrow
        assert nvd_schema.names == attack_schema.names


@pytest.mark.data_quality
class TestRequiredFields:
    @pytest.mark.parametrize("parquet_dir", [NVD_DIR, ATTACK_DIR])
    def test_no_nulls_in_required_fields(self, parquet_dir):
        _skip_if_missing(parquet_dir)
        required = ["record_id", "source_id", "source_record_id", "content", "raw", "ingested_at"]
        for f in parquet_dir.glob("*.parquet"):
            t = pq.ParquetFile(f).read()
            for col in required:
                nulls = pc.sum(pc.is_null(t[col])).as_py()
                assert nulls == 0, f"{f.name}: {col} has {nulls} nulls"


@pytest.mark.data_quality
class TestNVDInvariants:
    def test_record_ids_are_well_formed(self):
        _skip_if_missing(NVD_DIR)
        f = next(NVD_DIR.glob("*.parquet"))
        t = pq.ParquetFile(f).read()
        pattern = re.compile(r"^nvd:CVE-\d{4}-\d+$")
        ids = t["record_id"].to_pylist()
        bad = [i for i in ids if not pattern.match(i)]
        assert not bad, f"Malformed IDs: {bad[:5]}"
    
    def test_cvss_scores_in_valid_range(self):
        _skip_if_missing(NVD_DIR)
        f = next(NVD_DIR.glob("*.parquet"))
        t = pq.ParquetFile(f).read()
        scores = [s for s in t["cvss_score"].to_pylist() if s is not None]
        out_of_range = [s for s in scores if s < 0 or s > 10]
        assert not out_of_range, f"Out of range: {out_of_range[:5]}"
    
    def test_source_id_is_nvd(self):
        _skip_if_missing(NVD_DIR)
        f = next(NVD_DIR.glob("*.parquet"))
        t = pq.ParquetFile(f).read()
        unique = set(t["source_id"].to_pylist())
        assert unique == {"nvd"}


@pytest.mark.data_quality
class TestATTACKInvariants:
    def test_source_id_is_mitre_attack(self):
        _skip_if_missing(ATTACK_DIR)
        f = next(ATTACK_DIR.glob("enterprise-attack.parquet"))
        t = pq.ParquetFile(f).read()
        unique = set(t["source_id"].to_pylist())
        assert unique == {"mitre-attack"}


@pytest.mark.data_quality
class TestRawFidelity:
    @pytest.mark.parametrize("parquet_dir", [NVD_DIR, ATTACK_DIR])
    def test_raw_parses_back_to_valid_json(self, parquet_dir):
        _skip_if_missing(parquet_dir)
        import json
        f = next(parquet_dir.glob("*.parquet"))
        t = pq.ParquetFile(f).read()
        for row in t.to_pylist()[:500]:
            raw = json.loads(row["raw"])  # raises if invalid
            assert isinstance(raw, dict)
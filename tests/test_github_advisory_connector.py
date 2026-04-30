"""
Comprehensive tests for GitHubAdvisoryConnector.

Organized into six classes:
  TestIterRecords           — directory walking, ordering, edge cases
  TestNormalizeIdentification — record_id / source_id / source_record_id
  TestNormalizeContent      — content / title / fallback paths
  TestNormalizeVulnFields   — cve_id, severity, cwe_ids, cvss_score
  TestNormalizeMetadata     — ingested_at, published_at, license, raw
  TestNormalizeEndToEnd     — full fixture round-trip, uniqueness invariants
"""
import json
from datetime import timezone
from pathlib import Path

import pytest

from ingest.connectors.base import NormalizedData, VulnerabilityData
from ingest.connectors.github_advisory import GitHubAdvisoryConnector

FIXTURES = Path(__file__).parent / "fixtures" / "github-advisory"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_record(
    *,
    ghsa_id: str = "GHSA-xxxx-yyyy-zzzz",
    summary: str = "A test advisory",
    details: str = "Detailed description of the vulnerability.",
    aliases: list | None = None,
    severity_db: str = "HIGH",
    cwe_ids: list | None = None,
    severity_list: list | None = None,
    published: str = "2024-01-10T08:00:00Z",
) -> dict:
    """Build a minimal well-formed OSV advisory dict."""
    return {
        "schema_version": "1.4.0",
        "id": ghsa_id,
        "published": published,
        "modified": "2024-01-15T12:00:00Z",
        "aliases": aliases if aliases is not None else ["CVE-2024-99999"],
        "summary": summary,
        "details": details,
        "severity": severity_list if severity_list is not None else [],
        "affected": [],
        "references": [
            {"type": "ADVISORY", "url": f"https://github.com/advisories/{ghsa_id}"}
        ],
        "database_specific": {
            "cwe_ids": cwe_ids if cwe_ids is not None else ["CWE-502"],
            "severity": severity_db,
            "github_reviewed": True,
        },
    }


def _normalize(record: dict) -> VulnerabilityData:
    return GitHubAdvisoryConnector().normalize(record)


def _all_fixture_records() -> list[dict]:
    return list(GitHubAdvisoryConnector().iter_records(FIXTURES))


# ---------------------------------------------------------------------------
# TestIterRecords
# ---------------------------------------------------------------------------

class TestIterRecords:
    def test_yields_all_fixture_docs(self):
        """2 github-reviewed + 1 malware = 3 total."""
        records = _all_fixture_records()
        assert len(records) == 3

    def test_skips_unreviewed_directory(self, tmp_path):
        """unreviewed/ must never be walked even if present."""
        unreviewed = tmp_path / "advisories" / "unreviewed" / "GHSA-2024"
        unreviewed.mkdir(parents=True)
        (unreviewed / "GHSA-skip-skip-skip.json").write_text(
            json.dumps(_make_record(ghsa_id="GHSA-skip-skip-skip"))
        )
        records = list(GitHubAdvisoryConnector().iter_records(tmp_path))
        assert records == []

    def test_yields_nothing_from_empty_directory(self, tmp_path):
        records = list(GitHubAdvisoryConnector().iter_records(tmp_path))
        assert records == []

    def test_skips_missing_subdirs_gracefully(self, tmp_path):
        """A directory with no advisories/ subdir doesn't raise."""
        records = list(GitHubAdvisoryConnector().iter_records(tmp_path))
        assert records == []

    def test_handles_single_subdir_only(self, tmp_path):
        """Works when only github-reviewed/ exists (malware/ absent)."""
        reviewed = tmp_path / "advisories" / "github-reviewed" / "X"
        reviewed.mkdir(parents=True)
        (reviewed / "GHSA-test-test-test.json").write_text(
            json.dumps(_make_record(ghsa_id="GHSA-test-test-test"))
        )
        records = list(GitHubAdvisoryConnector().iter_records(tmp_path))
        assert len(records) == 1
        assert records[0]["id"] == "GHSA-test-test-test"

    def test_malware_records_are_included(self):
        """Malware-category advisories must appear in the output."""
        records = _all_fixture_records()
        ids = {r["id"] for r in records}
        assert "GHSA-mal1-mal2-mal3" in ids

    def test_all_yielded_records_have_id(self):
        for record in _all_fixture_records():
            assert "id" in record, f"Missing id in {record}"

    def test_skips_malformed_json_files(self, tmp_path):
        """A file with invalid JSON is silently skipped, not raised."""
        reviewed = tmp_path / "advisories" / "github-reviewed" / "X"
        reviewed.mkdir(parents=True)
        (reviewed / "bad.json").write_text("{invalid json{{")
        records = list(GitHubAdvisoryConnector().iter_records(tmp_path))
        assert records == []

    def test_multiple_nested_subdirectories(self, tmp_path):
        """Advisories are stored one level deep inside year-named dirs; rglob finds them."""
        for year, ghsa_id in [("2023", "GHSA-aaaa-0001-0001"), ("2024", "GHSA-bbbb-0002-0002")]:
            d = tmp_path / "advisories" / "github-reviewed" / year
            d.mkdir(parents=True)
            (d / f"{ghsa_id}.json").write_text(json.dumps(_make_record(ghsa_id=ghsa_id)))
        records = list(GitHubAdvisoryConnector().iter_records(tmp_path))
        assert len(records) == 2


# ---------------------------------------------------------------------------
# TestNormalizeIdentification
# ---------------------------------------------------------------------------

class TestNormalizeIdentification:
    def test_record_id_format(self):
        result = _normalize(_make_record(ghsa_id="GHSA-xxxx-yyyy-zzzz"))
        assert result.record_id == "github-advisory:GHSA-xxxx-yyyy-zzzz"

    def test_source_id_is_github_advisory(self):
        result = _normalize(_make_record())
        assert result.source_id == "github-advisory"

    def test_source_record_id_is_ghsa_id(self):
        result = _normalize(_make_record(ghsa_id="GHSA-xxxx-yyyy-zzzz"))
        assert result.source_record_id == "GHSA-xxxx-yyyy-zzzz"

    def test_source_url_uses_ghsa_id(self):
        result = _normalize(_make_record(ghsa_id="GHSA-xxxx-yyyy-zzzz"))
        assert result.source_url == "https://github.com/advisories/GHSA-xxxx-yyyy-zzzz"

    def test_source_url_is_none_when_id_missing(self):
        record = _make_record()
        record["id"] = ""
        result = _normalize(record)
        assert result.source_url is None


# ---------------------------------------------------------------------------
# TestNormalizeContent
# ---------------------------------------------------------------------------

class TestNormalizeContent:
    def test_details_maps_to_content(self):
        result = _normalize(_make_record(details="Full detailed description."))
        assert result.content == "Full detailed description."

    def test_summary_maps_to_title(self):
        result = _normalize(_make_record(summary="Short advisory title"))
        assert result.title == "Short advisory title"

    def test_falls_back_to_summary_when_details_empty(self):
        result = _normalize(_make_record(details="", summary="Fallback summary"))
        assert result.content == "Fallback summary"

    def test_falls_back_to_summary_when_details_absent(self):
        record = _make_record()
        del record["details"]
        result = _normalize(record)
        assert result.content == record["summary"]

    def test_content_is_empty_string_when_both_absent(self):
        record = _make_record(details="", summary="")
        result = _normalize(record)
        assert result.content == ""

    def test_title_is_none_when_summary_empty(self):
        record = _make_record(summary="")
        result = _normalize(record)
        assert result.title is None

    def test_content_hash_is_set(self):
        result = _normalize(_make_record(details="Some content"))
        assert result.content_hash is not None
        assert len(result.content_hash) == 64

    def test_content_length_is_positive_for_nonempty_content(self):
        result = _normalize(_make_record(details="Some content here"))
        assert result.content_length is not None
        assert result.content_length > 0

    def test_content_length_is_zero_for_empty_content(self):
        result = _normalize(_make_record(details="", summary=""))
        assert result.content_length == 0


# ---------------------------------------------------------------------------
# TestNormalizeVulnFields
# ---------------------------------------------------------------------------

class TestNormalizeVulnFields:
    def test_cve_alias_extracted(self):
        result = _normalize(_make_record(aliases=["CVE-2024-12345"]))
        assert result.cve_id == "CVE-2024-12345"

    def test_first_cve_alias_used_when_multiple(self):
        result = _normalize(_make_record(aliases=["CVE-2024-11111", "CVE-2024-22222"]))
        assert result.cve_id == "CVE-2024-11111"

    def test_cve_id_is_none_when_no_cve_alias(self):
        result = _normalize(_make_record(aliases=[]))
        assert result.cve_id is None

    def test_cve_id_is_none_when_only_non_cve_aliases(self):
        result = _normalize(_make_record(aliases=["GHSA-xxxx-yyyy-zzzz", "PYSEC-2024-1"]))
        assert result.cve_id is None

    def test_severity_critical_lowercased(self):
        result = _normalize(_make_record(severity_db="CRITICAL"))
        assert result.severity == "critical"

    def test_severity_high_lowercased(self):
        result = _normalize(_make_record(severity_db="HIGH"))
        assert result.severity == "high"

    def test_severity_moderate_lowercased(self):
        result = _normalize(_make_record(severity_db="MODERATE"))
        assert result.severity == "moderate"

    def test_severity_low_lowercased(self):
        result = _normalize(_make_record(severity_db="LOW"))
        assert result.severity == "low"

    def test_severity_is_none_when_absent(self):
        record = _make_record()
        record["database_specific"]["severity"] = ""
        result = _normalize(record)
        assert result.severity is None

    def test_cwe_ids_extracted(self):
        result = _normalize(_make_record(cwe_ids=["CWE-89", "CWE-502"]))
        assert result.cwe_ids == ["CWE-89", "CWE-502"]

    def test_cwe_ids_empty_list_when_absent(self):
        result = _normalize(_make_record(cwe_ids=[]))
        assert result.cwe_ids == []

    def test_cvss_score_none_when_severity_list_empty(self):
        result = _normalize(_make_record(severity_list=[]))
        assert result.cvss_score is None

    def test_cvss_score_parsed_when_plain_float_string(self):
        result = _normalize(_make_record(severity_list=[{"type": "CVSS_V3", "score": "7.5"}]))
        assert result.cvss_score == 7.5

    def test_cvss_score_none_for_vector_string(self):
        """CVSS vector strings (not bare scores) yield None — no vector math dep."""
        result = _normalize(_make_record(
            severity_list=[{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"}]
        ))
        assert result.cvss_score is None

    def test_cvss_v4_preferred_over_v3(self):
        result = _normalize(_make_record(severity_list=[
            {"type": "CVSS_V3", "score": "6.0"},
            {"type": "CVSS_V4", "score": "9.0"},
        ]))
        assert result.cvss_score == 9.0


# ---------------------------------------------------------------------------
# TestNormalizeMetadata
# ---------------------------------------------------------------------------

class TestNormalizeMetadata:
    def test_ingested_at_is_set(self):
        result = _normalize(_make_record())
        assert result.ingested_at is not None

    def test_ingested_at_is_utc(self):
        result = _normalize(_make_record())
        assert result.ingested_at.tzinfo == timezone.utc

    def test_published_at_parsed(self):
        result = _normalize(_make_record(published="2024-01-10T08:00:00Z"))
        assert result.published_at is not None
        assert result.published_at.year == 2024
        assert result.published_at.month == 1
        assert result.published_at.day == 10

    def test_published_at_is_none_when_absent(self):
        record = _make_record()
        del record["published"]
        result = _normalize(record)
        assert result.published_at is None

    def test_license_is_cc_by_4(self):
        result = _normalize(_make_record())
        assert result.license == "CC-BY-4.0"

    def test_returns_vulnerability_data_subclass(self):
        result = _normalize(_make_record())
        assert isinstance(result, VulnerabilityData)
        assert type(result) is VulnerabilityData

    def test_raw_contains_original_record(self):
        record = _make_record(ghsa_id="GHSA-xxxx-yyyy-zzzz")
        result = _normalize(record)
        assert result.raw is not None
        assert result.raw["id"] == "GHSA-xxxx-yyyy-zzzz"

    def test_raw_is_dict(self):
        result = _normalize(_make_record())
        assert isinstance(result.raw, dict)


# ---------------------------------------------------------------------------
# TestNormalizeEndToEnd
# ---------------------------------------------------------------------------

class TestNormalizeEndToEnd:
    def test_all_fixture_records_normalize_without_error(self):
        connector = GitHubAdvisoryConnector()
        for record in connector.iter_records(FIXTURES):
            result = connector.normalize(record)
            assert isinstance(result, VulnerabilityData)

    def test_all_normalized_record_ids_are_unique(self):
        connector = GitHubAdvisoryConnector()
        ids = [connector.normalize(r).record_id for r in connector.iter_records(FIXTURES)]
        assert len(ids) == len(set(ids)), f"Duplicate record_ids: {ids}"

    def test_all_normalized_records_have_source_id(self):
        connector = GitHubAdvisoryConnector()
        for record in connector.iter_records(FIXTURES):
            result = connector.normalize(record)
            assert result.source_id == "github-advisory"

    def test_all_normalized_records_have_non_empty_source_record_id(self):
        connector = GitHubAdvisoryConnector()
        for record in connector.iter_records(FIXTURES):
            result = connector.normalize(record)
            assert result.source_record_id, f"Empty source_record_id for {record}"

    def test_normalized_raw_is_json_serializable(self):
        import json as _json
        connector = GitHubAdvisoryConnector()
        for record in connector.iter_records(FIXTURES):
            result = connector.normalize(record)
            serialized = _json.dumps(result.raw)
            assert isinstance(_json.loads(serialized), dict)

    def test_fixture_with_cve_alias_has_cve_id(self):
        """GHSA-xxxx-yyyy-zzzz has CVE-2024-12345 in aliases."""
        connector = GitHubAdvisoryConnector()
        records = list(connector.iter_records(FIXTURES))
        target = next(r for r in records if r["id"] == "GHSA-xxxx-yyyy-zzzz")
        result = connector.normalize(target)
        assert result.cve_id == "CVE-2024-12345"

    def test_fixture_without_cve_alias_has_no_cve_id(self):
        """GHSA-aaaa-bbbb-cccc has no CVE alias."""
        connector = GitHubAdvisoryConnector()
        records = list(connector.iter_records(FIXTURES))
        target = next(r for r in records if r["id"] == "GHSA-aaaa-bbbb-cccc")
        result = connector.normalize(target)
        assert result.cve_id is None

    def test_malware_fixture_normalizes_correctly(self):
        connector = GitHubAdvisoryConnector()
        records = list(connector.iter_records(FIXTURES))
        target = next(r for r in records if r["id"] == "GHSA-mal1-mal2-mal3")
        result = connector.normalize(target)
        assert result.severity == "critical"
        assert result.cwe_ids == []
        assert "malware" in result.content.lower() or "exfiltrat" in result.content.lower()

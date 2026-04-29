"""
Comprehensive tests for BronConnector.

Organized into six classes:
  TestIterRecords        — file walking, ordering, edge cases
  TestNormalizeIdentification — record_id / source_id / source_record_id
  TestNormalizeContent   — content and title mapping, all fallback paths
  TestNormalizeSourceUrls — correct URL for every datatype + unknown
  TestNormalizeRaw       — ArangoDB internal field stripping, field preservation
  TestNormalizeMetadata  — ingested_at, language, unset security fields
  TestNormalizeEndToEnd  — full fixture round-trip, uniqueness invariants
"""
import json
from datetime import timezone
from pathlib import Path

import pytest

from ingest.connectors.base import NormalizedData
from ingest.connectors.bron import (
    BronConnector,
    _SOURCE_URL_TEMPLATES,
    _TARGET_COLLECTIONS,
)

FIXTURES = Path(__file__).parent / "fixtures" / "bron"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_record(
    *,
    original_id: str = "AML.M0000",
    name: str = "Test Entity",
    datatype: str = "atlas_mitigation",
    description: str = "A test description.",
    tags: list | None = None,
    include_arango_fields: bool = True,
    include_metadata: bool = True,
) -> dict:
    """Build a minimal well-formed BRON document dict."""
    record: dict = {
        "original_id": original_id,
        "name": name,
        "datatype": datatype,
    }
    if include_arango_fields:
        record["_id"] = f"{datatype}/{original_id}"
        record["_rev"] = "_testRev---"
        record["_key"] = original_id
    if include_metadata:
        record["metadata"] = {
            "description": description,
            "tags": tags if tags is not None else [],
        }
    return record


def _normalize(record: dict) -> NormalizedData:
    return BronConnector().normalize(record)


def _all_fixture_records() -> list[dict]:
    return list(BronConnector().iter_records(FIXTURES))


# ---------------------------------------------------------------------------
# TestIterRecords
# ---------------------------------------------------------------------------

class TestIterRecords:
    def test_yields_all_fixture_docs(self):
        """2 atlas_mitigation + 1 atlas_technique + 1 car = 4."""
        records = _all_fixture_records()
        assert len(records) == 4

    def test_collection_ordering_is_alphabetical(self):
        """Collections are walked in sorted order, so atlas_mitigation records
        come before atlas_technique, which come before car."""
        records = _all_fixture_records()
        datatypes = [r["datatype"] for r in records]
        # atlas_mitigation (x2) → atlas_technique → car
        assert datatypes == [
            "atlas_mitigation",
            "atlas_mitigation",
            "atlas_technique",
            "car",
        ]

    def test_skips_missing_collection_files_gracefully(self):
        """Collections in _TARGET_COLLECTIONS with no matching file on disk
        must be silently skipped, not raise FileNotFoundError."""
        records = _all_fixture_records()
        present_datatypes = {r["datatype"] for r in records}
        for collection in _TARGET_COLLECTIONS:
            fixture_file = FIXTURES / f"{collection}.json"
            if not fixture_file.exists():
                assert collection not in present_datatypes

    def test_yields_nothing_from_empty_directory(self, tmp_path):
        """A directory with no collection JSON files produces no records."""
        records = list(BronConnector().iter_records(tmp_path))
        assert records == []

    def test_handles_empty_collection_file(self, tmp_path):
        """A collection file containing an empty JSON array yields no records
        but does not raise."""
        (tmp_path / "atlas_mitigation.json").write_text("[]")
        records = list(BronConnector().iter_records(tmp_path))
        assert records == []

    def test_non_target_collection_file_is_ignored(self, tmp_path):
        """A JSON file whose name is not in _TARGET_COLLECTIONS (e.g. ATT&CK
        technique, which has its own connector) is never read."""
        (tmp_path / "technique.json").write_text(
            json.dumps([_make_record(datatype="technique")])
        )
        records = list(BronConnector().iter_records(tmp_path))
        assert records == []

    def test_all_yielded_records_have_required_keys(self):
        """Every record from iter_records must have original_id, name, datatype."""
        for record in _all_fixture_records():
            assert "original_id" in record, f"Missing original_id in {record}"
            assert "name" in record, f"Missing name in {record}"
            assert "datatype" in record, f"Missing datatype in {record}"

    def test_multiple_collections_in_same_directory(self, tmp_path):
        """Records from multiple collection files in one directory are all yielded."""
        (tmp_path / "atlas_mitigation.json").write_text(
            json.dumps([_make_record(original_id="AML.M0000", datatype="atlas_mitigation")])
        )
        (tmp_path / "car.json").write_text(
            json.dumps([_make_record(original_id="CAR-2013-01-001", datatype="car")])
        )
        records = list(BronConnector().iter_records(tmp_path))
        assert len(records) == 2
        assert records[0]["datatype"] == "atlas_mitigation"
        assert records[1]["datatype"] == "car"


# ---------------------------------------------------------------------------
# TestNormalizeIdentification
# ---------------------------------------------------------------------------

class TestNormalizeIdentification:
    def test_record_id_format(self):
        result = _normalize(_make_record(original_id="AML.M0000", datatype="atlas_mitigation"))
        assert result.record_id == "bron:atlas_mitigation:AML.M0000"

    def test_source_id_is_bron(self):
        result = _normalize(_make_record())
        assert result.source_id == "bron"

    def test_source_record_id_format(self):
        result = _normalize(_make_record(original_id="AML.M0000", datatype="atlas_mitigation"))
        assert result.source_record_id == "atlas_mitigation:AML.M0000"

    def test_record_id_includes_datatype(self):
        result = _normalize(_make_record(original_id="CAR-2013-01-001", datatype="car"))
        assert result.record_id == "bron:car:CAR-2013-01-001"

    def test_original_id_falls_back_to_key_when_missing(self):
        """When original_id is absent the connector must fall back to _key."""
        record = {
            "_id": "atlas_mitigation/AML.M0000",
            "_rev": "_testRev---",
            "_key": "AML.M0000",
            "name": "No original_id",
            "datatype": "atlas_mitigation",
            "metadata": {"description": "desc", "tags": []},
        }
        result = _normalize(record)
        assert result.record_id == "bron:atlas_mitigation:AML.M0000"
        assert result.source_record_id == "atlas_mitigation:AML.M0000"

    def test_original_id_takes_precedence_over_key(self):
        """If both original_id and _key are present, original_id wins."""
        record = _make_record(original_id="AML.M0000")
        record["_key"] = "DIFFERENT_KEY"
        result = _normalize(record)
        assert "AML.M0000" in result.record_id
        assert "DIFFERENT_KEY" not in result.record_id


# ---------------------------------------------------------------------------
# TestNormalizeContent
# ---------------------------------------------------------------------------

class TestNormalizeContent:
    def test_description_maps_to_content(self):
        record = _make_record(description="Specific description text.")
        result = _normalize(record)
        assert result.content == "Specific description text."

    def test_name_maps_to_title(self):
        record = _make_record(name="My Entity Name")
        result = _normalize(record)
        assert result.title == "My Entity Name"

    def test_falls_back_to_name_when_description_is_empty_string(self):
        record = _make_record(name="Fallback Name", description="")
        result = _normalize(record)
        assert result.content == "Fallback Name"

    def test_falls_back_to_name_when_description_key_absent(self):
        record = _make_record()
        del record["metadata"]["description"]
        result = _normalize(record)
        assert result.content == record["name"]

    def test_falls_back_to_name_when_metadata_is_absent(self):
        record = _make_record(include_metadata=False)
        result = _normalize(record)
        assert result.content == record["name"]

    def test_falls_back_to_name_when_metadata_is_explicit_none(self):
        record = _make_record()
        record["metadata"] = None
        result = _normalize(record)
        assert result.content == record["name"]

    def test_content_is_empty_string_when_both_description_and_name_are_absent(self):
        """Edge case: record with no name and no description yields empty content,
        not None — NormalizedData.content is a required str field."""
        record = {
            "_key": "X",
            "original_id": "X",
            "datatype": "atlas_mitigation",
            "metadata": {"description": "", "tags": []},
        }
        result = _normalize(record)
        assert result.content == ""
        assert result.title is None

    def test_description_in_real_fixture_is_non_empty(self):
        record = next(r for r in _all_fixture_records() if r["original_id"] == "AML.M0000")
        result = _normalize(record)
        assert len(result.content) > 0


# ---------------------------------------------------------------------------
# TestNormalizeSourceUrls
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("datatype,original_id,expected_url", [
    (
        "atlas_technique",
        "AML.T0000",
        "https://atlas.mitre.org/techniques/AML.T0000",
    ),
    (
        "atlas_tactic",
        "AML.TA0000",
        "https://atlas.mitre.org/tactics/AML.TA0000",
    ),
    (
        "atlas_mitigation",
        "AML.M0000",
        "https://atlas.mitre.org/mitigations/AML.M0000",
    ),
    (
        "car",
        "CAR-2013-01-001",
        "https://car.mitre.org/analytics/CAR-2013-01-001",
    ),
    (
        "d3fend_mitigation",
        "D3-AA",
        "https://d3fend.mitre.org/technique/D3-AA",
    ),
    (
        "engage_activity",
        "EAC0001",
        "https://engage.mitre.org/techniques/EAC0001",
    ),
    (
        "engage_approach",
        "EAP0001",
        "https://engage.mitre.org/approaches/EAP0001",
    ),
    (
        "engage_goal",
        "EGO0001",
        "https://engage.mitre.org/goals/EGO0001",
    ),
])
def test_source_url_for_each_datatype(datatype, original_id, expected_url):
    record = _make_record(original_id=original_id, datatype=datatype)
    result = _normalize(record)
    assert result.source_url == expected_url


def test_source_url_is_none_for_unknown_datatype():
    record = _make_record(datatype="unknown_collection")
    result = _normalize(record)
    assert result.source_url is None


def test_url_templates_cover_all_target_collections():
    """Every collection in _TARGET_COLLECTIONS must have a URL template."""
    missing = _TARGET_COLLECTIONS - set(_SOURCE_URL_TEMPLATES)
    assert not missing, f"Collections with no URL template: {missing}"


# ---------------------------------------------------------------------------
# TestNormalizeRaw
# ---------------------------------------------------------------------------

class TestNormalizeRaw:
    def test_strips_arango_id(self):
        result = _normalize(_make_record(include_arango_fields=True))
        assert "_id" not in result.raw

    def test_strips_arango_rev(self):
        result = _normalize(_make_record(include_arango_fields=True))
        assert "_rev" not in result.raw

    def test_strips_arango_key(self):
        result = _normalize(_make_record(include_arango_fields=True))
        assert "_key" not in result.raw

    def test_no_underscore_prefixed_keys_in_raw(self):
        result = _normalize(_make_record(include_arango_fields=True))
        underscore_keys = [k for k in result.raw if k.startswith("_")]
        assert underscore_keys == []

    def test_preserves_original_id(self):
        result = _normalize(_make_record(original_id="AML.M0000"))
        assert result.raw["original_id"] == "AML.M0000"

    def test_preserves_name(self):
        result = _normalize(_make_record(name="My Entity"))
        assert result.raw["name"] == "My Entity"

    def test_preserves_datatype(self):
        result = _normalize(_make_record(datatype="atlas_mitigation"))
        assert result.raw["datatype"] == "atlas_mitigation"

    def test_preserves_metadata_block(self):
        result = _normalize(_make_record(description="Some desc.", tags=["Evasion"]))
        assert result.raw["metadata"]["description"] == "Some desc."
        assert result.raw["metadata"]["tags"] == ["Evasion"]

    def test_raw_is_dict(self):
        result = _normalize(_make_record())
        assert isinstance(result.raw, dict)

    def test_raw_from_real_fixture_preserves_html_free_description(self):
        record = next(r for r in _all_fixture_records() if r["original_id"] == "AML.M0000")
        result = _normalize(record)
        assert result.raw["metadata"]["description"] == record["metadata"]["description"]


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

    def test_published_at_is_none(self):
        """BRON node documents carry no publication date."""
        result = _normalize(_make_record())
        assert result.published_at is None

    def test_produces_base_normalized_data_not_subclass(self):
        """BRON entities are general knowledge nodes, not vulnerabilities or
        detection rules, so the connector returns the base NormalizedData type."""
        from ingest.connectors.base import NormalizedData, VulnerabilityData
        result = _normalize(_make_record())
        assert type(result) is NormalizedData
        assert not isinstance(result, VulnerabilityData)


# ---------------------------------------------------------------------------
# TestNormalizeEndToEnd
# ---------------------------------------------------------------------------

class TestNormalizeEndToEnd:
    def test_all_fixture_records_normalize_without_error(self):
        """Every record produced by iter_records must normalize successfully."""
        connector = BronConnector()
        for record in connector.iter_records(FIXTURES):
            result = connector.normalize(record)
            assert isinstance(result, NormalizedData)

    def test_all_normalized_record_ids_are_unique(self):
        """record_id must be unique across all fixture records."""
        connector = BronConnector()
        ids = [
            connector.normalize(r).record_id
            for r in connector.iter_records(FIXTURES)
        ]
        assert len(ids) == len(set(ids)), f"Duplicate record_ids: {ids}"

    def test_all_normalized_records_have_non_empty_source_id(self):
        connector = BronConnector()
        for record in connector.iter_records(FIXTURES):
            result = connector.normalize(record)
            assert result.source_id == "bron"

    def test_all_normalized_records_have_non_empty_source_record_id(self):
        connector = BronConnector()
        for record in connector.iter_records(FIXTURES):
            result = connector.normalize(record)
            assert result.source_record_id, f"Empty source_record_id for {record}"

    def test_normalized_raw_is_json_serializable(self):
        """raw must be serializable since writers.py calls json.dumps on it."""
        import json as _json
        connector = BronConnector()
        for record in connector.iter_records(FIXTURES):
            result = connector.normalize(record)
            serialized = _json.dumps(result.raw)
            assert isinstance(_json.loads(serialized), dict)

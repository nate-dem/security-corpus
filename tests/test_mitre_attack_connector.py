import json
from pathlib import Path
from unittest.mock import patch

from ingest.connectors.base import NormalizedData, MitreData
from ingest.connectors.mitre_attack import MitreAttackConnector


SAMPLE_TECHNIQUE = json.loads(
    (Path(__file__).parent / "fixtures" / "mitre-attack" / "sample_technique.json").read_text()
)


def test_normalize_returns_mitre_data():
    result = MitreAttackConnector().normalize(SAMPLE_TECHNIQUE)
    assert isinstance(result, MitreData)
    assert isinstance(result, NormalizedData)
    assert result.record_id.startswith("mitre-attack:")
    assert result.source_id == "mitre-attack"


def test_normalize_maps_name_and_description():
    result = MitreAttackConnector().normalize(SAMPLE_TECHNIQUE)
    assert result.title == SAMPLE_TECHNIQUE["name"]
    assert result.content == SAMPLE_TECHNIQUE["description"]


def test_iter_records_filters_revoked_and_deprecated():
    objects = [
        SAMPLE_TECHNIQUE,
        {**SAMPLE_TECHNIQUE, "id": "attack-pattern--revoked", "revoked": True},
        {**SAMPLE_TECHNIQUE, "id": "attack-pattern--deprecated", "x_mitre_deprecated": True},
        {"type": "relationship", "id": "relationship--1234"},
    ]
    bundle_path = Path("fake.json")
    connector = MitreAttackConnector()
    with patch("ingest.connectors.mitre_attack.read", return_value=iter(objects)):
        results = list(connector.iter_records(bundle_path))
    assert len(results) == 1
    assert results[0]["id"] == SAMPLE_TECHNIQUE["id"]


def test_normalize_populates_new_fields():
    result = MitreAttackConnector().normalize(SAMPLE_TECHNIQUE)
    assert result.content_hash is not None
    assert result.content_length is not None
    assert result.content_length > 0
    assert result.license is not None
    assert result.framework == "attack"
    assert result.category_id == "T1055.011"



from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from ingest.connectors.base import NormalizedData
from ingest.readers import read


# Collections unique to BRON — excludes technique/tactic/capec/cwe/cve which
# are already covered by dedicated connectors.
_TARGET_COLLECTIONS = frozenset({
    "atlas_technique",
    "atlas_tactic",
    "atlas_mitigation",
    "car",
    "d3fend_mitigation",
    "engage_activity",
    "engage_approach",
    "engage_goal",
})

_SOURCE_URL_TEMPLATES: dict[str, str] = {
    "atlas_technique": "https://atlas.mitre.org/techniques/{id}",
    "atlas_tactic": "https://atlas.mitre.org/tactics/{id}",
    "atlas_mitigation": "https://atlas.mitre.org/mitigations/{id}",
    "car": "https://car.mitre.org/analytics/{id}",
    "d3fend_mitigation": "https://d3fend.mitre.org/technique/{id}",
    "engage_activity": "https://engage.mitre.org/techniques/{id}",
    "engage_approach": "https://engage.mitre.org/approaches/{id}",
    "engage_goal": "https://engage.mitre.org/goals/{id}",
}


class BronConnector:
    source_id = "bron"

    def iter_records(self, path: Path) -> Iterator[dict]:
        """Yield one document per node from each target BRON collection.

        path is a directory containing {collection}.json files, each a JSON
        array exported from the BRON ArangoDB instance via export_bron.py.
        Missing collection files are silently skipped.
        """
        for collection in sorted(_TARGET_COLLECTIONS):
            collection_path = path / f"{collection}.json"
            if not collection_path.exists():
                continue
            yield from read(collection_path, json_path="item")

    def normalize(self, record: dict) -> NormalizedData:
        """Convert one BRON node document into the normalized schema."""
        original_id = record.get("original_id") or record.get("_key", "")
        datatype = record.get("datatype", "")
        metadata = record.get("metadata") or {}

        content = metadata.get("description", "") or record.get("name", "")

        url_template = _SOURCE_URL_TEMPLATES.get(datatype)
        source_url = url_template.format(id=original_id) if url_template else None

        # Strip ArangoDB internal fields (_id, _rev, _key) from the stored raw.
        raw = {k: v for k, v in record.items() if not k.startswith("_")}

        return NormalizedData(
            record_id=f"bron:{datatype}:{original_id}",
            source_id=self.source_id,
            source_record_id=f"{datatype}:{original_id}",
            content=content,
            title=record.get("name"),
            raw=raw,
            ingested_at=datetime.now(timezone.utc),
            source_url=source_url,
        )

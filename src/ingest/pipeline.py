from pathlib import Path
from typing import Iterator

from ingest.connectors.base import Connector, NormalizedData
from ingest.connectors.capec import CapecConnector
from ingest.connectors.cisa_kev import CisaKevConnector
from ingest.connectors.mitre_attack import MitreAttackConnector
from ingest.connectors.nvd import NVDConnector
from ingest.writers import write_parquet


_CONNECTORS: dict[str, Connector] = {
    "nvd": NVDConnector(),
    "mitre-attack": MitreAttackConnector(),
    "cisa-kev": CisaKevConnector(),
    "capec": CapecConnector(),
}

def ingest(path: Path, source: str) -> Iterator[NormalizedData]:
    """Stream normalized records from a file for a known source."""
    connector = _CONNECTORS.get(source)
    if connector is None:
        raise ValueError(f"{source} is not a valid source in {list(_CONNECTORS)}")
    
    for record in connector.iter_records(path):
        yield connector.normalize(record)

def ingest_and_store(path: Path, source: str, output_dir: Path) -> int:
    """Ingest a file end-to-end and write normalized records to Parquet.
    
    Returns the number of records written.
    """
    records = ingest(path, source)
    return write_parquet(records, output_dir, source=source, input_path=path)
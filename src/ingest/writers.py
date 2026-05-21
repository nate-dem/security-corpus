from pathlib import Path
from typing import Iterable
import json
from datetime import date, datetime

import pyarrow as pa
import pyarrow.parquet as pq

from ingest.connectors.base import NormalizedData

_STRIPPED_SUFFIXES = [".json", ".gz", ".xml", ".zip", ".zst"]

def write_parquet(
    normalized_records: Iterable[NormalizedData],
    output_dir: Path,
    source: str,
    input_path: Path,
) -> int:
    """Write normalized data records to a Parquet file under output_dir.

    File layout: {output_dir}/source_id={source}/{input_filename}.parquet
    Returns the number of records written.
    """
    rows = [_record_to_row(r) for r in normalized_records]

    if not rows:
        return 0

    stem = input_path.name
    suffixes = []

    # collect all suffixes that need to be stripped
    for suffix in input_path.suffixes:
        if suffix in stem:
            suffixes.append(suffix)

    # remove all found suffixes in reverse order so that each element is actually the current suffix when it's checked
    for suffix in reversed(suffixes):
        if suffix in _STRIPPED_SUFFIXES:
            stem = stem.removesuffix(suffix)

    output_path = output_dir / f"source_id={source}" / f"{stem}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_path, compression="snappy")

    return len(rows)

def _json_default(obj):
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def _record_to_row(normalized_record: NormalizedData) -> dict:
    """Convert NormalizedData (or subclass) into a Parquet-friendly dict."""
    data = normalized_record.model_dump()
    # Serialize raw dict as JSON string if present, otherwise leave as None
    if data.get("raw") is not None:
        data["raw"] = json.dumps(data["raw"], default=_json_default)
    return data

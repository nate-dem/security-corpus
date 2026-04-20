import gzip
import json
from pathlib import Path
from typing import Any, Iterator

import ijson


def read(path: Path, json_path: str | None = None) -> Iterator[Any]:
    """Stream parsed objects from a file. Currently supports the following file types:
        - .json.gz
        - .json
    
    For JSON files, `json_path` is an ijson path expression (e.g.
    'vulnerabilities.item' in NVD CVE data) selecting which array to stream records from.
    If omitted, the entire document is yielded as a single object.
    """
    suffixes = path.suffixes
    
    if suffixes[-2:] == [".json", ".gz"]:
        yield from _read_json_gz(path, json_path)
    elif suffixes[-1:] == [".json"]:
        yield from _read_json(path, json_path)
    else:
        raise ValueError(
            f"Unsupported format: {path.name} (suffixes={suffixes})"
        )

def _read_json_gz(path: Path, json_path: str | None) -> Iterator[Any]:
    with gzip.open(path, "rb") as f:
        yield from _stream_json(f, json_path)

def _read_json(path: Path, json_path: str | None) -> Iterator[Any]:
    with open(path, "rb") as f:
        yield from _stream_json(f, json_path)

def _stream_json(fp, json_path: str | None) -> Iterator[Any]:
    if json_path:
        yield from ijson.items(fp, json_path)
    else:
        yield json.load(fp)
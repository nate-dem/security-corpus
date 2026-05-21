import gzip
import io
import json
from pathlib import Path
from typing import Any, Iterator

import ijson
import yaml
from lxml import etree


def read(path: Path, json_path: str | None = None, xml_tag: str | None = None) -> Iterator[Any]:
    """Stream parsed objects from a file. Currently supports the following file types:
        - .json.gz
        - .json
        - .xml
        - .yml / .yaml
        - .zst (zstandard-compressed JSONL)
    """
    suffixes = path.suffixes

    if suffixes[-2:] == [".json", ".gz"]:
        yield from _read_json_gz(path, json_path)
    elif suffixes[-1:] == [".json"]:
        yield from _read_json(path, json_path)
    elif suffixes[-1:] == [".xml"]:
        yield from _read_xml(path, xml_tag)
    elif suffixes[-1:] in ([".yml"], [".yaml"]):
        yield from _read_yaml(path)
    elif suffixes[-1:] == [".zst"]:
        yield from _read_jsonl_zst(path)
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
        yield from ijson.items(fp, json_path, use_float=True)
    else:
        yield json.load(fp)


def _read_yaml(path: Path) -> Iterator[Any]:
    with open(path, "rb") as f:
        yield yaml.safe_load(f)


def _read_jsonl_zst(path: Path) -> Iterator[Any]:
    """Stream JSON objects from a zstandard-compressed JSONL file."""
    import zstandard as zstd

    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _read_xml(path: Path, xml_tag: str | None) -> Iterator[Any]:
    if not xml_tag:
        raise ValueError("xml_tag is required for XML files")

    context = etree.iterparse(str(path), events=("end",), tag=xml_tag)
    for _event, elem in context:
        yield elem
        # free memory: clear this element and drop parent references
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
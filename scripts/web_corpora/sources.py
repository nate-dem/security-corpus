"""Approved source snapshots; these are download choices, not inclusion filters."""

SOURCES = {
    "redsage-cfw": {
        "repo_id": "RISys-Lab/RedSage-CFW",
        "revision": "52b393cb49c59789c5cae272acc0dbc58c0831ff",
        "suffix": ".parquet",
    },
    "primus-fineweb": {
        "repo_id": "trendmicro-ailab/Primus-FineWeb",
        "revision": "41058dbfa82fd0b11a12605cad17139ca57558d4",
        "suffix": ".jsonl.gz",
    },
}


def data_files(manifest: dict) -> list[dict]:
    suffix = SOURCES[manifest["source"]]["suffix"]
    return [entry for entry in manifest["files"] if entry["path"].endswith(suffix)]

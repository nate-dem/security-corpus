"""Export target BRON collections from the public ArangoDB instance to JSON files.

Usage:
    python scripts/export_bron.py

Writes one JSON array file per collection to data/bron/raw/.
Run this once before running ingest_bron.py.
"""
import base64
import json
import urllib.request
from pathlib import Path

_BRON_BASE = "http://bron.alfa.csail.mit.edu:8529/_db/BRON"
# Public read-only instance uses the ArangoDB guest user with no password.
_AUTH_HEADER = {"Authorization": "Basic " + base64.b64encode(b"guest:guest").decode()}

_TARGET_COLLECTIONS = [
    "atlas_technique",
    "atlas_tactic",
    "atlas_mitigation",
    "car",
    "d3fend_mitigation",
    "engage_activity",
    "engage_approach",
    "engage_goal",
]


def _fetch_collection(collection: str) -> list[dict]:
    """Fetch all documents from a BRON collection using the ArangoDB cursor API."""
    payload = json.dumps({
        "query": f"FOR doc IN {collection} RETURN doc",
        "batchSize": 1000,
    }).encode()

    req = urllib.request.Request(
        f"{_BRON_BASE}/_api/cursor",
        data=payload,
        headers={"Content-Type": "application/json", **_AUTH_HEADER},
    )
    with urllib.request.urlopen(req) as resp:
        body = json.loads(resp.read())

    docs: list[dict] = list(body["result"])

    # Paginate through remaining batches if the result set is large.
    while body.get("hasMore") and body.get("id"):
        cursor_url = f"{_BRON_BASE}/_api/cursor/{body['id']}"
        req = urllib.request.Request(
            cursor_url,
            data=b"",
            method="PUT",
            headers={"Content-Type": "application/json", **_AUTH_HEADER},
        )
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read())
        docs.extend(body["result"])

    return docs


def main() -> None:
    output_dir = Path("data/bron/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    for collection in _TARGET_COLLECTIONS:
        print(f"Fetching {collection}...", end=" ", flush=True)
        try:
            docs = _fetch_collection(collection)
        except Exception as exc:
            print(f"FAILED ({exc})")
            continue
        out_path = output_dir / f"{collection}.json"
        out_path.write_text(json.dumps(docs, indent=2))
        print(f"{len(docs)} docs → {out_path}")


if __name__ == "__main__":
    main()

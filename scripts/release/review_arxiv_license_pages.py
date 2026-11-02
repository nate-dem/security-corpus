"""Save and inspect official article license declarations for an explicit ID list.

This records current upstream declarations, not permission or a match between
the retained text and a downloaded paper version. No corpus data is changed.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

from lxml import html
import requests

from scripts.release.reconcile_arxiv_rights import canonical_license


def parse_page(payload, identifier):
    tree = html.fromstring(payload)
    ids = tree.xpath('//meta[@name="citation_arxiv_id"]/@content')
    if ids != [identifier]:
        raise ValueError(f"Article identity mismatch: {ids}")
    urls = tree.xpath(
        '//div[contains(concat(" ", normalize-space(@class), " "),'
        ' " abs-license ")]//a/@href'
    )
    if len(urls) != 1:
        raise ValueError(f"Expected one article license, found {urls}")
    license_value = canonical_license(urls[0])
    if not license_value or license_value.startswith("UNRECOGNIZED:"):
        raise ValueError(f"Unrecognized explicit license: {urls[0]}")
    versions = tree.xpath('//meta[@property="og:url"]/@content')
    version = (
        re.search(r"/abs/" + re.escape(identifier) + r"(v\d+)$", versions[0])
        if versions
        else None
    )
    if version is None:
        raise ValueError("Missing version-specific article identity")
    history = tree.xpath('//div[@class="submission-history"]//text()')
    return dict(
        license=license_value,
        license_url=urls[0],
        article_version=identifier + version.group(1),
        versions_in_history=sorted(
            set(re.findall(r"\[v(\d+)\]", " ".join(history))), key=int
        ),
        title=tree.xpath('//meta[@name="citation_title"]/@content')[0],
        paper_version_binding_established=False,
        public_release_approved=False,
        license_basis="official_article_page_declaration_version_unbound",
    )


def run(targets, output):
    output.mkdir(parents=True, exist_ok=True)
    pages = output / "pages"
    pages.mkdir(exist_ok=True)
    rows = [json.loads(line) for line in targets.read_text().splitlines()]
    results = []
    session = requests.Session()
    session.headers["User-Agent"] = (
        "security-corpus-license-review/1.0 (metadata research)"
    )
    last_request = 0.0
    for index, row in enumerate(rows, 1):
        identifier = row["record_id"].removeprefix("arxiv:")
        if not re.fullmatch(r"\d{4}\.\d{4,5}", identifier):
            raise ValueError("Unsupported identifier")
        requested_version = row.get("review_version")
        if requested_version is not None and (
            type(requested_version) is not int or requested_version < 1
        ):
            raise ValueError("Invalid requested paper version")
        requested_id = identifier + (
            f"v{requested_version}" if requested_version else ""
        )
        path = pages / f"{requested_id}.html"
        note_path = pages / f"{requested_id}.json"
        url = "https://arxiv.org/abs/" + requested_id
        try:
            if path.exists() and note_path.exists():
                payload = path.read_bytes()
                note = json.loads(note_path.read_text())
                if hashlib.sha256(payload).hexdigest() != note["sha256"]:
                    raise ValueError("Cached evidence changed")
            else:
                time.sleep(max(0, 3.2 - (time.monotonic() - last_request)))
                last_request = time.monotonic()
                response = session.get(url, timeout=30)
                response.raise_for_status()
                payload = response.content
                parse_page(payload, identifier)
                path.write_bytes(payload)
                note = dict(
                    url=url,
                    response_url=response.url,
                    retrieved_at=datetime.now(timezone.utc).isoformat(),
                    sha256=hashlib.sha256(payload).hexdigest(),
                )
                note_path.write_text(json.dumps(note, indent=2) + "\n")
            parsed = parse_page(payload, identifier)
            if requested_version and parsed["article_version"] != requested_id:
                raise ValueError("Requested version differs from returned version")
            result = dict(
                record_id=row["record_id"],
                content_hash=row["content_hash"],
                file=row["file"],
                row=row["row"],
                previous_license=row["license"],
                evidence_path=str(path.resolve()),
                **note,
                **parsed,
                quality_observation=row.get("quality_observation"),
                status="declaration_found",
            )
        except (requests.RequestException, ValueError, IndexError) as error:
            result = dict(
                record_id=row["record_id"],
                url=url,
                status="unresolved",
                error=str(error),
            )
        results.append(result)
        (output / "records.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in results)
        )
        if index % 10 == 0 or result["status"] != "declaration_found":
            print(f"{index}/{len(rows)} {result['status']}", flush=True)
    summary = dict(
        checked_at=datetime.now(timezone.utc).isoformat(),
        requested=len(rows),
        checked=len(results),
        statuses=dict(Counter(r["status"] for r in results)),
        licenses=dict(Counter(r["license"] for r in results if "license" in r)),
        paper_version_binding_established=False,
        public_release_approved=False,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.targets, args.output), indent=2))

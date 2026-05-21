from pathlib import Path
from typing import Iterator

from ingest.connectors.base import Connector, NormalizedData
from ingest.connectors.detection import SigmaConnector
from ingest.connectors.knowledge import (
    BronConnector,
    CapecConnector,
    MitreAttackConnector,
    MitreCweConnector,
)
from ingest.connectors.logs import CloudTrailSessionConnector
from ingest.connectors.stackexchange import StackExchangeSiteConnector
from ingest.connectors.reddit import RedditSubredditConnector, REDDIT_SUBREDDITS
from ingest.connectors.arxiv import ArxivConnector
from ingest.connectors.transcripts import YouTubeTranscriptsConnector
from ingest.connectors.vulnerability import (
    CisaKevConnector,
    GitHubAdvisoryConnector,
    NVDConnector,
)
from ingest.writers import write_parquet


_CONNECTORS: dict[str, Connector] = {
    "nvd": NVDConnector(),
    "mitre-attack": MitreAttackConnector(),
    "cisa-kev": CisaKevConnector(),
    "capec": CapecConnector(),
    "mitre-cwe": MitreCweConnector(),
    "sigma": SigmaConnector(),
    "stackexchange-infosec": StackExchangeSiteConnector("infosec", "security.stackexchange.com"),
    "stackexchange-reverseengineering": StackExchangeSiteConnector("reverseengineering", "reverseengineering.stackexchange.com"),
    "stackexchange-crypto": StackExchangeSiteConnector("crypto", "crypto.stackexchange.com"),
    "stackexchange-tor": StackExchangeSiteConnector("tor", "tor.stackexchange.com"),
    "bron": BronConnector(),
    "github-advisory": GitHubAdvisoryConnector(),
    "youtube-transcripts": YouTubeTranscriptsConnector(),
    "cloudtrail-flaws": CloudTrailSessionConnector(),
    "arxiv": ArxivConnector(),
}

for _sub in REDDIT_SUBREDDITS:
    _CONNECTORS[f"reddit-{_sub.lower()}"] = RedditSubredditConnector(_sub)

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

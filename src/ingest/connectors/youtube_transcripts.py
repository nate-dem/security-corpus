"""Compatibility alias for the moved YouTube transcripts connector."""

import sys

from ingest.connectors.transcripts import youtube_transcripts as _impl

sys.modules[__name__] = _impl


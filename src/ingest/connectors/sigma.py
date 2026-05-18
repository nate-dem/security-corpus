"""Compatibility alias for the moved Sigma connector."""

import sys

from ingest.connectors.detection import sigma as _impl

sys.modules[__name__] = _impl


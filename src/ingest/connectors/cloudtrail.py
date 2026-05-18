"""Compatibility alias for the moved CloudTrail connector."""

import sys

from ingest.connectors.logs import cloudtrail as _impl

sys.modules[__name__] = _impl


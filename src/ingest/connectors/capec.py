"""Compatibility alias for the moved CAPEC connector."""

import sys

from ingest.connectors.knowledge import capec as _impl

sys.modules[__name__] = _impl


"""Compatibility alias for the moved BRON connector."""

import sys

from ingest.connectors.knowledge import bron as _impl

sys.modules[__name__] = _impl


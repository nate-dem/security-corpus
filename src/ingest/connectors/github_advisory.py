"""Compatibility alias for the moved GitHub Advisory connector."""

import sys

from ingest.connectors.vulnerability import github_advisory as _impl

sys.modules[__name__] = _impl


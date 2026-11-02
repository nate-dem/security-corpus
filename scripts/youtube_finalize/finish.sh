#!/usr/bin/env bash
set -euo pipefail
YT_FINISH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python3 "$YT_FINISH_ROOT/scripts/youtube_finalize/submit.py" "$@"

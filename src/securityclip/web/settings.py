"""Runtime settings for the Security Scope web app."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from securityclip.config import default_index_dir


@dataclass(frozen=True)
class WebSettings:
    index_dir: Path
    web_db: Path
    router_model: str
    planner_model: str
    synthesis_model: str
    max_commands: int
    # max_results is the display/retrieval cap (documents shown per query), not the
    # synthesis payload cap. The synthesis_* fields below bound what the LLM receives.
    max_results: int
    max_limit: int
    max_head_count: int
    command_timeout_seconds: float
    max_output_chars: int
    openai_api_key_present: bool
    # Synthesis payload caps — decoupled from retrieval so a broad query (many large
    # documents) cannot produce an LLM request big enough to trigger a proxy 413.
    synthesis_max_sources_per_root: int = 8
    synthesis_max_citations_per_root: int = 12
    synthesis_max_output_chars_per_operation: int = 2000
    synthesis_max_total_chars: int = 50_000


def load_settings() -> WebSettings:
    index_dir = default_index_dir()
    web_db = Path(os.environ.get("SECURITYCLIP_WEB_DB", str(index_dir / "securityclip_web.sqlite"))).expanduser()
    return WebSettings(
        index_dir=index_dir,
        web_db=web_db,
        router_model=os.environ.get("SECURITYCLIP_ROUTER_MODEL", "gpt-5-nano"),
        planner_model=os.environ.get("SECURITYCLIP_PLANNER_MODEL", "gpt-5-mini"),
        synthesis_model=os.environ.get("SECURITYCLIP_SYNTHESIS_MODEL", "gpt-5-mini"),
        max_commands=_env_int("SECURITYCLIP_MAX_COMMANDS", 8),
        max_results=_env_int("SECURITYCLIP_MAX_RESULTS", 50),
        max_limit=_env_int("SECURITYCLIP_MAX_LIMIT", 100),
        max_head_count=_env_int("SECURITYCLIP_MAX_HEAD_COUNT", 200),
        command_timeout_seconds=float(os.environ.get("SECURITYCLIP_COMMAND_TIMEOUT", "60")),
        max_output_chars=_env_int("SECURITYCLIP_MAX_OUTPUT_CHARS", 50_000),
        openai_api_key_present=bool(os.environ.get("OPENAI_API_KEY")),
        synthesis_max_sources_per_root=_env_int("SECURITYCLIP_SYNTHESIS_MAX_SOURCES_PER_ROOT", 8),
        synthesis_max_citations_per_root=_env_int("SECURITYCLIP_SYNTHESIS_MAX_CITATIONS_PER_ROOT", 12),
        synthesis_max_output_chars_per_operation=_env_int("SECURITYCLIP_SYNTHESIS_MAX_OUTPUT_CHARS_PER_OPERATION", 2000),
        synthesis_max_total_chars=_env_int("SECURITYCLIP_SYNTHESIS_MAX_TOTAL_CHARS", 50_000),
    )


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default

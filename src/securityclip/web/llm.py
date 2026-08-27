"""OpenAI-backed JSON planning and synthesis helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol


class ModelClient(Protocol):
    def generate_json(self, *, model: str, system: str, user: str) -> Any:
        ...

    def generate_text(self, *, model: str, system: str, user: str) -> str:
        ...


@dataclass
class OpenAIModelClient:
    """Thin wrapper around the OpenAI Responses API."""

    def _client(self):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError("openai package is required for model calls") from exc
        return OpenAI()

    def generate_json(self, *, model: str, system: str, user: str) -> Any:
        text = self.generate_text(
            model=model,
            system=system,
            user=f"{user}\n\nReturn only valid JSON. Do not wrap it in Markdown.",
        )
        return parse_json_text(text)

    def generate_text(self, *, model: str, system: str, user: str) -> str:
        response = self._client().responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        output_text = getattr(response, "output_text", None)
        if output_text:
            return str(output_text)
        return str(response)


def parse_json_text(text: str) -> Any:
    stripped = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start_candidates = [idx for idx in (stripped.find("{"), stripped.find("[")) if idx >= 0]
        if not start_candidates:
            raise
        start = min(start_candidates)
        end = max(stripped.rfind("}"), stripped.rfind("]"))
        if end < start:
            raise
        return json.loads(stripped[start : end + 1])


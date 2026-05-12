from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Bucket:
    name: str
    path: Path
    token_count: int
    description: str
    validation_sets: list[str] = field(default_factory=list)


class BucketRegistry:
    """
    Loads bucket definitions from YAML and tracks live token counts.

    Token counts start at 0 in the YAML; script 01_prepare_buckets.py
    writes them back after tokenization so downstream code sees real sizes.
    """

    def __init__(self, config_path: str | Path):
        cfg_path = Path(config_path)
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        self._cfg_path = cfg_path
        self.buckets: dict[str, Bucket] = {}
        for b in cfg["buckets"]:
            self.buckets[b["name"]] = Bucket(
                name=b["name"],
                path=Path(b["path"]),
                token_count=int(b["token_count"]),
                description=b["description"],
                validation_sets=b.get("validation_sets", []),
            )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def names(self) -> list[str]:
        return list(self.buckets.keys())

    def token_counts(self) -> dict[str, int]:
        return {n: b.token_count for n, b in self.buckets.items()}

    def total_tokens(self) -> int:
        return sum(b.token_count for b in self.buckets.values())

    def natural_weights(self) -> dict[str, float]:
        """Weight proportional to each bucket's token count."""
        total = self.total_tokens()
        if total == 0:
            count = len(self.buckets)
            return {name: 1.0 / count for name in self.names()}
        return {name: b.token_count / total for name, b in self.buckets.items()}

    def uniform_weights(self) -> dict[str, float]:
        n = len(self.buckets)
        return {name: 1.0 / n for name in self.names()}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def update_token_count(self, bucket_name: str, count: int) -> None:
        """Write an updated token count back to the YAML config."""
        self.buckets[bucket_name].token_count = count
        with open(self._cfg_path) as f:
            cfg = yaml.safe_load(f)
        for b in cfg["buckets"]:
            if b["name"] == bucket_name:
                b["token_count"] = count
                break
        with open(self._cfg_path, "w") as f:
            yaml.dump(cfg, f, sort_keys=False)

    def stats(self) -> str:
        lines = [f"{'Bucket':<30} {'Tokens':>15} {'Share':>8}"]
        lines.append("-" * 55)
        total = self.total_tokens()
        for name, bucket in self.buckets.items():
            share = bucket.token_count / total if total else 0
            lines.append(f"{name:<30} {bucket.token_count:>15,} {share:>7.1%}")
        lines.append("-" * 55)
        lines.append(f"{'TOTAL':<30} {total:>15,} {'100.0%':>8}")
        return "\n".join(lines)

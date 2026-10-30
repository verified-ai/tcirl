"""Turn a metric or run key into a filesystem-safe name."""

from __future__ import annotations


def safe_filename(key: str) -> str:
    return key.replace("/", "_")

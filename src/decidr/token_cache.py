"""Speculative token-boundary cache. See docs/SPEC.md §9 for the
normative design: what it stores, how it's used speculatively within
`Client._decide_prefix`, and why it's always verified against the real
response, never trusted blindly."""

from __future__ import annotations

import json
import os
from pathlib import Path


def default_path() -> Path:
    return Path.home() / ".decidr" / "token-cache.json"


class TokenCache:
    """Maps (model, option_id) -> the real token sequence observed the
    last time that option's id was fully, non-speculatively resolved.
    Used to speculatively fire the next round's request before the
    current round's response even arrives, gated on live confirmation
    every step -- see SPEC.md §9.3. A wrong prediction only ever wastes
    one extra request; it never produces a wrong `Decision`, since every
    speculative guess is checked against the real response before being
    trusted."""

    def __init__(self, path: str | Path | None = None, persist: bool = True):
        self.persist = persist
        self.path = Path(path) if path is not None else default_path()
        self._data: dict[str, dict[str, list[str]]] = {}
        self._dirty = False
        if self.persist:
            self._data = self._load()

    def _load(self) -> dict[str, dict[str, list[str]]]:
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def get(self, model: str, option_id: str) -> list[str] | None:
        return self._data.get(model, {}).get(option_id)

    def set(self, model: str, option_id: str, tokens: list[str]) -> None:
        self._data.setdefault(model, {})[option_id] = tokens
        self._dirty = True

    def save(self) -> None:
        """Flush to disk if anything changed and persistence is enabled.
        Safe to call liberally -- a cache that can't be written to disk
        still works in-memory for the rest of this process."""
        if not self._dirty or not self.persist:
            return
        self._dirty = False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(f"{self.path.suffix}.tmp{os.getpid()}")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f)
            tmp.replace(self.path)
        except OSError:
            pass

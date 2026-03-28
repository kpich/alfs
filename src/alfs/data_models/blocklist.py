"""Persistent blocklist of word forms that should never get dictionary entries.

The blocklist is stored as a human-editable YAML file mapping form → reason.
It is sorted alphabetically for clean version-control diffs.

Usage:
    from alfs.data_models.blocklist import Blocklist

    bl = Blocklist(Path("../alfs_data/blocklist.yaml"))
    bl.add("thrumbly", reason="tokenization artifact")
    if bl.contains("thrumbly"):
        ...
"""

from __future__ import annotations

from pathlib import Path

import yaml


class Blocklist:
    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> dict[str, str | None]:
        """Return {form: reason_or_None}. Empty dict if file absent."""
        if not self._path.exists():
            return {}
        raw = yaml.safe_load(self._path.read_text()) or {}
        return {str(k): (str(v) if v is not None else None) for k, v in raw.items()}

    def save(self, entries: dict[str, str | None]) -> None:
        """Atomically write blocklist (sorted alphabetically)."""
        sorted_entries = dict(sorted(entries.items()))
        tmp = self._path.with_suffix(".yaml.tmp")
        tmp.write_text(yaml.dump(sorted_entries, allow_unicode=True, sort_keys=False))
        tmp.rename(self._path)

    def contains(self, form: str) -> bool:
        return form in self.load()

    def add(self, form: str, reason: str | None = None) -> None:
        entries = self.load()
        if form not in entries:
            entries[form] = reason
            self.save(entries)

    def add_many(self, forms: list[str], reason: str | None = None) -> int:
        """Add multiple forms. Returns count newly added."""
        entries = self.load()
        added = 0
        for form in forms:
            if form not in entries:
                entries[form] = reason
                added += 1
        if added:
            self.save(entries)
        return added

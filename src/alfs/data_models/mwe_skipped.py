"""Persistent list of MWE candidate forms that have been reviewed and skipped.

Forms in this list are excluded from future enqueue runs, preventing duplicate
review work. Stored as a sorted YAML list.

Usage:
    from alfs.data_models.mwe_skipped import MWESkipped

    s = MWESkipped(Path("../alfs_data/mwe_skipped.yaml"))
    s.add("the president")
    forms = s.load()
"""

from __future__ import annotations

from pathlib import Path

import yaml


class MWESkipped:
    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> list[str]:
        """Return list of skipped forms. Empty list if file absent."""
        if not self._path.exists():
            return []
        raw = yaml.safe_load(self._path.read_text()) or []
        return [str(f) for f in raw]

    def save(self, forms: list[str]) -> None:
        """Atomically write skipped forms (sorted alphabetically)."""
        tmp = self._path.with_suffix(".yaml.tmp")
        tmp.write_text(yaml.dump(sorted(forms), allow_unicode=True))
        tmp.rename(self._path)

    def add(self, form: str) -> None:
        forms = self.load()
        if form not in forms:
            forms.append(form)
            self.save(forms)

    def add_many(self, forms: list[str]) -> int:
        """Add multiple forms. Returns count newly added."""
        existing = set(self.load())
        new: list[str] = []
        seen: set[str] = set()
        for f in forms:
            if f not in existing and f not in seen:
                new.append(f)
                seen.add(f)
        if new:
            self.save(list(existing) + new)
        return len(new)

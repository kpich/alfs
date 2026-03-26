"""Tests for dedupe.find_candidates()."""

from alfs.data_models.alf import Alf, Sense
from alfs.update.refinement.dedupe import find_candidates


def _alf(form: str, redirect: str | None = None) -> Alf:
    senses = [] if redirect else [Sense(definition="a definition")]
    return Alf(form=form, senses=senses, redirect=redirect)


def test_find_candidates_detects_case_variants() -> None:
    entries = {
        "Run": _alf("Run"),
        "run": _alf("run"),
        "cat": _alf("cat"),
    }
    candidates = find_candidates(entries)
    assert ("Run", "run") in candidates
    assert len(candidates) == 1


def test_find_candidates_excludes_existing_redirects() -> None:
    entries = {
        "Colour": _alf("Colour", redirect="color"),  # already a redirect
        "colour": _alf("colour"),
        "Color": _alf("Color"),
        "color": _alf("color", redirect="Color"),  # lower is redirect
    }
    candidates = find_candidates(entries)
    # "Colour" is already a redirect — skip
    # "color" is already a redirect — skip "Color"
    assert len(candidates) == 0


def test_find_candidates_requires_lowercase_form_to_exist() -> None:
    entries = {
        "Fox": _alf("Fox"),
        # "fox" not in entries
    }
    candidates = find_candidates(entries)
    assert len(candidates) == 0

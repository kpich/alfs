# Python

- Always run Python via `uv run` (e.g. `uv run python -m alfs.foo`), not `python3` directly.

# Testing

- Test files: `[module]_test.py`, co-located with the module (not in a separate `tests/` dir)
- Test functions: root-level `def test_[property]()` (no classes)
- Runner: pytest (`make test`)
- Dev deps (includes pytest): `make dev`

# User-invocable skills

- `/cc-induction` — Induce new word senses from CC task files in `../cc_tasks/pending/`
- `/cc-rewrite` — Rewrite sense definitions from CC task files in `../cc_tasks/pending/`
- `/cc-trim` — Trim redundant senses from CC task files in `../cc_tasks/pending/`
- `/cc-morph` — Identify morphological redirects from CC task files in `../cc_tasks/pending/`

# Python

- Always run Python via `uv run` (e.g. `uv run python -m alfs.foo`), not `python3` directly.

# Testing

- Test files: `[module]_test.py`, co-located with the module (not in a separate `tests/` dir)
- Test functions: root-level `def test_[property]()` (no classes)
- Runner: pytest (`make test`)
- Dev deps (includes pytest): `make dev`

# Skills

- Skill files are always local to this repo (never in `~/.claude`). Do not search `~` for skill files.

# User-invocable skills

- `/cc-induction` — Induce new word senses from CC task files in `../cc_tasks/pending/`
- `/cc-rewrite` — Rewrite sense definitions from CC task files in `../cc_tasks/pending/`
- `/cc-trim` — Trim redundant senses from CC task files in `../cc_tasks/pending/`
- `/cc-qc` — Per-form quality control: morph_rel links, sense deletion/rewriting/POS correction, spelling variant marking, case normalization, entry deletion+blocklist
- `/cc-spelling-variant` — Confirm spelling variant pairs from CC task files in `../cc_tasks/pending/`

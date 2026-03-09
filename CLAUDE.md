# Testing

- Test files: `[module]_test.py`, co-located with the module (not in a separate `tests/` dir)
- Test functions: root-level `def test_[property]()` (no classes)
- Runner: pytest (`make test`)
- Dev deps (includes pytest): `make dev`

# User-invocable skills

- cc-induction: `src/alfs/cc/skills/induction.md` — Induce new word senses from CC task files in `../cc_tasks/pending/`
- cc-rewrite: `src/alfs/cc/skills/rewrite.md` — Rewrite sense definitions from CC task files in `../cc_tasks/pending/`
- cc-trim: `src/alfs/cc/skills/trim_sense.md` — Trim redundant senses from CC task files in `../cc_tasks/pending/`
- cc-morph: `src/alfs/cc/skills/morph_redirect.md` — Identify morphological redirects from CC task files in `../cc_tasks/pending/`

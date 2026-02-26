# Testing

- Test files: `[module]_test.py`, co-located with the module (not in a separate `tests/` dir)
- Test functions: root-level `def test_[property]()` (no classes)
- Runner: pytest (`make test`)
- Dev deps (includes pytest): `make dev`

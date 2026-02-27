in dev stay tuned

## Prerequisites

- [ollama](https://ollama.com) — pull the model before running `make update`:
  ```
  ollama pull llama3.1:8b
  ```

## Make targets

| Target | What it does |
|---|---|
| `make dev` | Install dev dependencies (pytest, mypy, etc.) |
| `make etl` | Download + parse wikibooks dump → `text_data/` |
| `make seg` | Segment docs into word-form occurrences → `seg_data/` |
| `make update` | Induce senses + label occurrences → `alfs_data/`, `update_data/` |
| `make test` | Run tests |
| `make mypy` | Type-check |

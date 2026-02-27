in dev stay tuned

## Prerequisites

- [ollama](https://ollama.com) — pull the model before running `make update`:
  ```
  ollama pull llama3.1:8b
  ```
- Run `make compile` before `make viewer` to generate `viewer_data/data.json`.

## Make targets

| Target | What it does |
|---|---|
| `make dev` | Install dev dependencies (pytest, mypy, etc.) |
| `make etl` | Download + parse wikibooks dump → `text_data/` |
| `make seg` | Segment docs into word-form occurrences → `seg_data/` |
| `make update` | Induce senses + label occurrences → `alfs_data/` (`alfs.json`, `labeled.parquet`), `update_data/` (per-run archive) |
| `make relabel` | Re-label all occurrences for every word in `alfs.json` from scratch, replacing `alfs_data/labeled.parquet` |
| `make compile` | Compile viewer data from alfs.json + labeled occurrences → `viewer_data/data.json` |
| `make viewer`  | Start local Flask viewer at http://localhost:5001 |
| `make test` | Run tests |
| `make mypy` | Type-check |

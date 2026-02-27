in dev stay tuned

## Prerequisites

- [ollama](https://ollama.com): `ollama pull gemma2:9b`
- [nextflow](https://nextflow.io)
- Run `make compile` before `make viewer`

## Make targets

| Target | What it does |
|---|---|
| `make dev` | Install dev dependencies |
| `make etl` | Download + parse wikibooks dump → `text_data/` |
| `make seg` | Segment docs → `seg_data/` |
| `make update` | Induce senses + label occurrences → `alfs_data/` |
| `make relabel` | Re-label all occurrences from scratch |
| `make compile` | Compile → `viewer_data/data.json` |
| `make viewer` | Start viewer at http://localhost:5001 |
| `make test` | Run tests |
| `make mypy` | Type-check |

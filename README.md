in dev stay tuned

## Prerequisites

- [ollama](https://ollama.com): `ollama pull gemma2:9b`
- [nextflow](https://nextflow.io)
- Run `make compile` before `make viewer`

## Make targets

| Target | What it does |
|---|---|
| `make dev` | Install dev dependencies |
| `make etl` | Download + parse text data → `text_data/` |
| `make seg` | Segment docs → `seg_data/` |
| `make update` | Induce senses + label occurrences → `alfs_data/` |
| `make relabel [NWORDS=N]` | Re-label N randomly-selected words (default 5); set NWORDS larger for broader runs |
| `make dedupe` | LLM-check case-variant entries → set redirects in `alfs.json` |
| `make postag` | LLM-assign POS tags to all untagged senses in `alfs.json` |
| `make validate` | Check that all byte_offsets in `labeled.parquet` still resolve to the right token in `docs.parquet`; exits nonzero if any are stale |
| `make compile` | Compile → `viewer_data/data.json` |
| `make viewer` | Start viewer at http://localhost:5001 |
| `make conductor` | Start work-queue conductor UI at http://localhost:5002 |
| `make test` | Run tests |
| `make mypy` | Type-check |

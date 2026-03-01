in dev stay tuned

## Prerequisites

- [ollama](https://ollama.com): `ollama pull gemma2:9b`
- [nextflow](https://nextflow.io)

## Make targets

### Preprocessing

| Target | What it does |
|---|---|
| `make dev` | Install dev dependencies |
| `make etl` | Download + parse text data → `text_data/` |
| `make seg` | Segment docs → `seg_data/` |

### Conductor (main entrypoint)

| Target | What it does |
|---|---|
| `make conductor` | Start work-queue UI at http://localhost:5002 |
| `make update` | Induce senses + label occurrences → `alfs_data/` |
| `make relabel [NWORDS=N]` | Re-label N randomly-selected words (default 5) |
| `make dedupe` | LLM-check case-variant entries → set redirects in `senses.db` |
| `make postag` | LLM-assign POS tags to all untagged senses in `senses.db` |
| `make cleanup` | Clear stale senses from redirect entries in `senses.db` |
| `make rewrite` | LLM-suggest definition rewrites for 5 random words → `changes.db` `[H]` |
| `make retag` | LLM-re-evaluate POS tags for 10 random tagged words → `changes.db` `[H]` |
| `make prune` | Queue removal of top-5 worst-quality senses (>20% rated <3) → `changes.db` `[H]` |
| `make queenant` | Start human-approval UI for rewrite/retag/prune changes at http://localhost:5003 |

### Viewer

| Target | What it does |
|---|---|
| `make compile` | Compile → `viewer_data/data.json` |
| `make viewer` | Start viewer at http://localhost:5001 |

### Dev

| Target | What it does |
|---|---|
| `make test` | Run tests |
| `make mypy` | Type-check |

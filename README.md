in dev stay tuned

## Prerequisites

- [ollama](https://ollama.com): `ollama pull gemma2:9b`
- [nextflow](https://nextflow.io)

## Two long-running processes

Keep these running in separate terminals during active development:

```
make conductor   # work-queue UI at http://localhost:5002
make clerk-watch # apply queued sense mutations from clerk_queue/
```

`make conductor` dispatches pipeline jobs (update, relabel, etc.).
`make clerk-watch` picks up the sense-mutation requests those jobs enqueue and applies them to `senses.db` / `labeled.db`. Without it, changes accumulate in `clerk_queue/pending/` but are never applied.

(Alternatively, run `make clerk` without watch mode to drain the queue once and exit.)

## Make targets

### Preprocessing

| Target | What it does |
|---|---|
| `make dev` | Install dev dependencies |
| `make etl` | Download + parse text data → `text_data/` |
| `make seg` | Segment docs → `seg_data/` |

### Conductor + Clerk (main entrypoints)

| Target | What it does |
|---|---|
| `make conductor` | Start work-queue UI at http://localhost:5002 |
| `make clerk-watch` | Start clerk worker in watch mode (persistent) |
| `make clerk` | Drain clerk queue once and exit |
| `make update` | Induce senses + label occurrences → `alfs_data/` |
| `make relabel [NWORDS=N]` | Re-label N randomly-selected words (default 5) |
| `make label_new [NWORDS=N]` | Label new instances for N random inventory words (default 10) |
| `make dedupe` | LLM-check case-variant entries → enqueue redirects |
| `make postag` | LLM-assign POS tags to all untagged senses → enqueue updates |
| `make cleanup` | Clear stale senses from redirect entries in `senses.db` |
| `make rewrite` | LLM-suggest definition rewrites for 5 random words → enqueue |
| `make retag` | LLM-re-evaluate POS tags for 10 random tagged words → enqueue |
| `make prune` | Queue removal of top-5 worst-quality senses (>20% rated <3) |
| `make morph_redirect` | Queue morphological redirect proposals (e.g. plural, past tense) |
| `make trim_senses` | LLM-identify redundant senses in 50 random multi-sense words → enqueue |

### Viewer

| Target | What it does |
|---|---|
| `make compile` | Compile → `viewer_data/data.json` |
| `make viewer` | Start viewer at http://localhost:5001 |

### Backup

| Target | What it does |
|---|---|
| `make backup` | Commit senses YAML to `alfs_senses/` repo |
| `make backup-gdrive` | Sync `text_data/`, `alfs_data/`, `seg_data/` to Google Drive via rclone |

`make backup-gdrive` requires a one-time setup:

```bash
brew install rclone
rclone config   # New remote → name "gdrive" → Google Drive → follow prompts → browser auth
```

The remote must be named `gdrive` (or override with `make backup-gdrive GDRIVE_REMOTE=myremote`). Files are synced to `alfs_backup/` on Google Drive, with `text_data/cache/`, SQLite WAL files, and `latest` symlinks excluded.

### Dev

| Target | What it does |
|---|---|
| `make test` | Run tests |
| `make mypy` | Type-check |

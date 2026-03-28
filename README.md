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

### Groq batch labeling

An alternative to the local-LLM labeling pipeline that uses Groq's batch API — cheaper and faster for large runs.

**1. Prepare the batch files**

```
make groq-batch-prepare [GROQ_MODEL=llama-3.1-8b-instant]
```

Writes timestamped files to `../groq_batch/`. If the request count exceeds Groq's 50k limit, multiple chunks are created automatically:
- `batch_input_YYYYMMDDTHHMMSS_001.jsonl` — upload this to Groq
- `batch_metadata_YYYYMMDDTHHMMSS_001.jsonl` — sidecar (do not move; ingest finds it automatically)
- `batch_input_YYYYMMDDTHHMMSS_002.jsonl`, `batch_metadata_YYYYMMDDTHHMMSS_002.jsonl`, etc. if needed

**2. Submit each chunk to Groq and download the results**

Upload each `batch_input_*.jsonl` file to Groq's console or batch API. Each is a separate batch job (≤50k requests). When a job completes, download the output file and save it anywhere in `../groq_batch/` — the filename doesn't matter.

**3. Ingest each output**

```
make groq-batch-ingest BATCH_OUTPUT=../groq_batch/<downloaded_filename>.jsonl
```

The matching metadata file is auto-discovered from `../groq_batch/` by matching request IDs. After a successful ingest, all three files (input, metadata, output) are moved to `../groq_batch_archive/` for safekeeping. Repeat for each chunk.

`backup-gdrive` syncs `../groq_batch_archive/` to Google Drive so all batch history is preserved.

---

### Claude Code (CC) mode

When the "CC" toggle is active in the conductor, pipeline tasks write JSON task files to `cc_tasks/pending/` instead of calling a local LLM. You then run CC skills (`/cc-induction`, `/cc-rewrite`, `/cc-trim`, `/cc-morph`) in Claude Code to process them, and apply the results via `make cc_apply`.

| Target | What it does |
|---|---|
| `make cc_apply` | Convert CC skill outputs in `cc_tasks/done/` into clerk requests |
| `make cc-clean` | Remove all pending and done CC task files |

### Viewer

| Target | What it does |
|---|---|
| `make compile` | Compile → `viewer_data/data.json` |
| `make viewer` | Start dictionary viewer at http://localhost:5001 |
| `make dataviewer` | Start ETL corpus viewer at http://localhost:5003 |

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

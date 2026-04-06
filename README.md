in dev stay tuned

## Prerequisites

- [ollama](https://ollama.com): `ollama pull qwen2.5:32b` (required for local LLM induction)
- [nextflow](https://nextflow.io)

## Two long-running processes

Keep these running in separate terminals during active development:

```
make conductor   # work-queue UI at http://localhost:5002
make clerk-watch # apply queued sense mutations from clerk_queue/
```

`make conductor` dispatches pipeline jobs.
`make clerk-watch` picks up the sense-mutation requests those jobs enqueue and applies them to `senses.db` / `labeled.db`. Without it, changes accumulate in `clerk_queue/pending/` but are never applied.

(Alternatively, run `make clerk` without watch mode to drain the queue once and exit.)

## Make targets

### Preprocessing

| Target | What it does |
|---|---|
| `make dev` | Install dev dependencies |
| `make etl` | Download + parse text data → `text_data/` |
| `make seg` | Segment docs → `seg_data/` |

### Main workflow

The core loop: enqueue forms, induce senses, apply clerk mutations.

| Target | What it does |
|---|---|
| `make enqueue_new_forms` | Find top-N unseen corpus forms by frequency → add to `induction_queue.yaml` |
| `make enqueue_poor_coverage` | Find forms with poorly-labeled occurrences → add to `induction_queue.yaml` |
| `make induce_senses` | Dequeue forms and run induction via local LLM (qwen2.5:32b via ollama) |
| `make cc_induce_senses` | Dequeue forms and write CC task files to `cc_tasks/pending/induction/` |
| `make clerk` | Drain clerk queue once and exit |
| `make clerk-watch` | Start clerk worker in watch mode (persistent) |
| `make postag` | LLM-assign POS tags to all untagged senses → enqueue updates |

Key variables:
- `INDUCTION_QUEUE` (default: `../alfs_data/induction_queue.yaml`)
- `BLOCKLIST_FILE` (default: `../alfs_data/blocklist.yaml`)
- `ENQUEUE_TOP_N` (default: 50)
- `ENQUEUE_MIN_COUNT` (default: 5)

### Claude Code (CC) mode

`make cc_induce_senses` writes task files to `cc_tasks/pending/` instead of calling the local LLM. Run the `/cc-induction` skill in Claude Code to process them, then apply results:

| Target | What it does |
|---|---|
| `make cc_apply` | Convert CC skill outputs in `cc_tasks/done/` into clerk requests + occurrence labels |
| `make cc-clean` | Remove all pending and done CC task files |

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

### Critic batch pass

Second-pass quality check: groups labeled instances by sense, shows them to a critic LLM, and downgrades incorrectly-labeled ones to `rating=0` (making them eligible for re-labeling). Each instance gains `last_critic_date`/`last_critic_model` fields; only unreviewed or stale instances are included in new batches.

```
make critic-batch-prepare [CRITIC_MODEL=openai/gpt-oss-20b]
# → ../critic_batch/critic_input_*.jsonl  (upload to Groq)
#   ../critic_batch/critic_metadata_*.jsonl  (sidecar, keep in place)

make critic-batch-ingest BATCH_OUTPUT=../critic_batch/<output>.jsonl
# → updates labeled.db; archives files to ../critic_batch_archive/
```

### Viewer

| Target | What it does |
|---|---|
| `make compile` | Compile → `viewer_data/data.json` |
| `make viewer` | Start dictionary viewer at http://localhost:5001 |
| `make dataviewer` | Start ETL corpus viewer at http://localhost:5003 |

### Backup

| Target | What it does |
|---|---|
| `make backup` | Commit senses YAML + blocklist/queue files to `alfs_senses/` repo |
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

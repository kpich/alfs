.PHONY: download etl seg enqueue_new_forms enqueue_poor_coverage induce_senses cc_induce_senses postag validate compile viewer dataviewer backup backup-gdrive conductor clerk clerk-watch cc_apply cc_morphrel_block cc-clean install_precommit_hooks dev test mypy cleandata groq-batch-prepare groq-batch-ingest

SENSES_DB          ?= ../alfs_data/senses.db
LABELED_DB         ?= ../alfs_data/labeled.db
CLERK_QUEUE        ?= ../clerk_queue
TEXT_DATA_DIR      ?= ../text_data
CACHE_DIR          ?= $(TEXT_DATA_DIR)/cache
NGRAM_CACHE        ?= $(TEXT_DATA_DIR)/ngram_cache.npy
DOCS               ?= $(TEXT_DATA_DIR)/docs.parquet
SENSES_REPO        ?= ../alfs_senses
SEG_DATA_DIR       ?= ../seg_data/by_prefix
GDRIVE_REMOTE      ?= gdrive
GDRIVE_DEST        ?= alfs_backup
NWORDS             ?= 5
SENSE_UPDATE_MODEL ?= qwen2.5:32b
LABEL_MODEL        ?= gemma2:9b
CC_TASKS_DIR       ?= ../cc_tasks
GROQ_BATCH_DIR     ?= ../groq_batch
GROQ_ARCHIVE_DIR   ?= ../groq_batch_archive
GROQ_MODEL         ?= llama-3.1-8b-instant
INSTANCE_LOG       ?= ../alfs_data/instance_log
SOURCE             ?= wikibooks
N_DOCS             ?= 10000
INDUCTION_QUEUE    ?= ../alfs_data/induction_queue.yaml
BLOCKLIST_FILE     ?= ../alfs_data/blocklist.yaml
ENQUEUE_TOP_N      ?= 500
ENQUEUE_MIN_COUNT  ?= 5
ENQUEUE_N_OCC_REFS ?= 3
CC_MORPHREL_N      ?= 20

download:
	uv run --no-sync python -m alfs.etl.download \
		--source $(SOURCE) --cache-dir $(CACHE_DIR)

etl:
	uv run --no-sync python -m alfs.etl.augment \
		--source $(SOURCE) --corpus $(DOCS) \
		--cache-dir $(CACHE_DIR) --ngram-cache $(NGRAM_CACHE) --n-docs $(N_DOCS)

seg:
	bash scripts/seg.sh \
		--params.docs $(DOCS) \
		--params.seg_data_dir $(SEG_DATA_DIR)

enqueue_new_forms:
	uv run --no-sync python -m alfs.update.induction.enqueue_new_forms \
		--seg-data-dir $(SEG_DATA_DIR) \
		--senses-db $(SENSES_DB) \
		--queue-file $(INDUCTION_QUEUE) \
		--blocklist-file $(BLOCKLIST_FILE) \
		--top-n $(ENQUEUE_TOP_N) \
		--min-count $(ENQUEUE_MIN_COUNT) \
		--n-occurrence-refs $(ENQUEUE_N_OCC_REFS)

enqueue_poor_coverage:
	uv run --no-sync python -m alfs.update.induction.enqueue_poor_coverage \
		--labeled-db $(LABELED_DB) \
		--queue-file $(INDUCTION_QUEUE) \
		--blocklist-file $(BLOCKLIST_FILE) \
		--top-n $(ENQUEUE_TOP_N)

induce_senses:
	uv run --no-sync python -m alfs.update.induction.induce_senses \
		--queue-file $(INDUCTION_QUEUE) \
		--blocklist-file $(BLOCKLIST_FILE) \
		--seg-data-dir $(SEG_DATA_DIR) \
		--docs $(DOCS) \
		--senses-db $(SENSES_DB) \
		--labeled-db $(LABELED_DB) \
		--queue-dir $(CLERK_QUEUE) \
		--model $(SENSE_UPDATE_MODEL)

cc_induce_senses:
	uv run --no-sync python -m alfs.update.induction.induce_senses \
		--queue-file $(INDUCTION_QUEUE) \
		--blocklist-file $(BLOCKLIST_FILE) \
		--seg-data-dir $(SEG_DATA_DIR) \
		--docs $(DOCS) \
		--senses-db $(SENSES_DB) \
		--labeled-db $(LABELED_DB) \
		--queue-dir $(CLERK_QUEUE) \
		--cc-tasks-dir $(CC_TASKS_DIR)

postag:
	uv run --no-sync python -m alfs.update.refinement.postag \
		--senses-db $(SENSES_DB) \
		--labeled-db $(LABELED_DB) \
		--docs $(DOCS) \
		--queue-dir $(CLERK_QUEUE) \
		--model $(SENSE_UPDATE_MODEL)

clerk:
	uv run --no-sync python -m alfs.clerk.worker \
		--queue-dir $(CLERK_QUEUE) \
		--senses-db $(SENSES_DB) \
		--labeled-db $(LABELED_DB) \
		--instance-log $(INSTANCE_LOG)

cc_apply:
	uv run --no-sync python -m alfs.cc.apply \
		--cc-tasks-dir $(CC_TASKS_DIR) \
		--senses-db $(SENSES_DB) \
		--queue-dir $(CLERK_QUEUE) \
		--labeled-db $(LABELED_DB) \
		--blocklist-file $(BLOCKLIST_FILE)

cc_morphrel_block:
	uv run --no-sync python -m alfs.update.refinement.generate_morphrel_block_tasks \
		--senses-db $(SENSES_DB) \
		--cc-tasks-dir $(CC_TASKS_DIR) \
		--blocklist-file $(BLOCKLIST_FILE) \
		--n $(CC_MORPHREL_N)

cc-clean:
	rm -f $(CC_TASKS_DIR)/pending/*/*.json $(CC_TASKS_DIR)/done/*/*.json

clerk-watch:
	uv run --no-sync python -m alfs.clerk.worker \
		--queue-dir $(CLERK_QUEUE) \
		--senses-db $(SENSES_DB) \
		--labeled-db $(LABELED_DB) \
		--watch \
		--instance-log $(INSTANCE_LOG)

validate:
	uv run --no-sync python -m alfs.qc.validate_labels \
		--labeled-db $(LABELED_DB) --docs $(DOCS)

compile:
	bash scripts/compile.sh

viewer:
	bash scripts/viewer.sh

dataviewer:
	bash scripts/dataviewer.sh --docs $(DOCS)

backup:
	uv run --no-sync python -m alfs.backup \
		--senses-db $(SENSES_DB) --senses-repo $(SENSES_REPO) \
		--queue-dir $(CLERK_QUEUE) \
		--blocklist-file $(BLOCKLIST_FILE) \
		--queue-file $(INDUCTION_QUEUE)

backup-gdrive:
	rclone sync ../text_data $(GDRIVE_REMOTE):$(GDRIVE_DEST)/text_data \
		--exclude "cache/**" \
		--progress
	rclone sync ../alfs_data $(GDRIVE_REMOTE):$(GDRIVE_DEST)/alfs_data \
		--exclude "*.db-wal" \
		--exclude "*.db-shm" \
		--progress
	rclone sync ../seg_data $(GDRIVE_REMOTE):$(GDRIVE_DEST)/seg_data \
		--progress
	rclone sync $(GROQ_ARCHIVE_DIR) $(GDRIVE_REMOTE):$(GDRIVE_DEST)/groq_batch_archive \
		--progress

conductor:
	uv run --no-sync python -m alfs.anthill

install_precommit_hooks:
	uv sync --group dev
	uv run pre-commit install

dev:
	uv sync --group dev

test:
	uv run pytest

mypy:
	uv run mypy src/

groq-batch-prepare:
	uv run --no-sync python -m alfs.update.labeling.groq_batch_prepare \
		--senses-db $(SENSES_DB) --labeled-db $(LABELED_DB) \
		--seg-data-dir $(SEG_DATA_DIR) --docs $(DOCS) \
		--output-dir $(GROQ_BATCH_DIR) --model $(GROQ_MODEL)

groq-batch-ingest:
	uv run --no-sync python -m alfs.update.labeling.groq_batch_ingest \
		--batch-output $(BATCH_OUTPUT) \
		--batch-dir $(GROQ_BATCH_DIR) \
		--senses-db $(SENSES_DB) --labeled-db $(LABELED_DB) \
		--log-dir $(INSTANCE_LOG) \
		--archive-dir $(GROQ_ARCHIVE_DIR)

cleandata:
	@echo "This will delete: ../alfs_data  ../seg_data  ../update_data  ../viewer_data"
	@echo "text_data (ETL output) will NOT be touched."
	@read -p "Continue? [y/N] " ans && [ "$$ans" = "y" ] || { echo "Aborted."; exit 1; }
	rm -rf ../alfs_data ../seg_data ../update_data ../viewer_data

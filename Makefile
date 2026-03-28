.PHONY: download etl seg update relabel label_new dedupe postag cleanup rewrite retag prune spelling_variant morph_redirect undo_morph trim_senses delete_entry validate compile viewer dataviewer backup backup-gdrive conductor clerk clerk-watch cc_apply cc-clean install_precommit_hooks dev test mypy cleandata groq-batch-prepare groq-batch-ingest

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

download:
	uv run --no-sync python -m alfs.etl.download \
		--source $(SOURCE) --cache-dir $(CACHE_DIR)

etl:
	uv run --no-sync python -m alfs.etl.augment \
		--source $(SOURCE) --corpus $(DOCS) \
		--cache-dir $(CACHE_DIR) --ngram-cache $(NGRAM_CACHE) --n-docs $(N_DOCS)

seg:
	uv run --no-sync python -m alfs.seg.augment \
		--docs $(DOCS) --seg-data-dir $(SEG_DATA_DIR)

update:
	bash scripts/update.sh \
		--seg-data-dir $(SEG_DATA_DIR) \
		--docs $(DOCS) \
		--senses-db $(SENSES_DB) \
		--labeled-db $(LABELED_DB) \
		--queue-dir $(CLERK_QUEUE) \
		--model $(SENSE_UPDATE_MODEL)

relabel:
	bash scripts/relabel.sh \
		--senses-db $(SENSES_DB) \
		--labeled-db $(LABELED_DB) \
		--docs $(DOCS) \
		--seg-data-dir $(SEG_DATA_DIR) \
		--nwords $(NWORDS) \
		--model $(LABEL_MODEL) \
		--log-dir $(INSTANCE_LOG)

label_new:
	bash scripts/label_new.sh \
		--senses-db $(SENSES_DB) \
		--labeled-db $(LABELED_DB) \
		--docs $(DOCS) \
		--seg-data-dir $(SEG_DATA_DIR) \
		--nwords $(NWORDS) \
		--model $(LABEL_MODEL) \
		--log-dir $(INSTANCE_LOG)

dedupe:
	uv run --no-sync python -m alfs.update.refinement.dedupe \
		--senses-db $(SENSES_DB) \
		--queue-dir $(CLERK_QUEUE) \
		--model $(SENSE_UPDATE_MODEL)

postag:
	uv run --no-sync python -m alfs.update.refinement.postag \
		--senses-db $(SENSES_DB) \
		--labeled-db $(LABELED_DB) \
		--docs $(DOCS) \
		--queue-dir $(CLERK_QUEUE) \
		--model $(SENSE_UPDATE_MODEL)

cleanup:
	uv run --no-sync python -m alfs.update.refinement.cleanup \
		--senses-db $(SENSES_DB) \
		--queue-dir $(CLERK_QUEUE)

rewrite:
	uv run --no-sync python -m alfs.update.refinement.rewrite \
		--senses-db $(SENSES_DB) \
		--queue-dir $(CLERK_QUEUE) \
		--model $(SENSE_UPDATE_MODEL)

retag:
	uv run --no-sync python -m alfs.update.refinement.retag \
		--senses-db $(SENSES_DB) \
		--labeled-db $(LABELED_DB) \
		--docs $(DOCS) \
		--queue-dir $(CLERK_QUEUE) \
		--model $(SENSE_UPDATE_MODEL)

prune:
	uv run --no-sync python -m alfs.update.refinement.prune \
		--senses-db $(SENSES_DB) \
		--labeled-db $(LABELED_DB) \
		--queue-dir $(CLERK_QUEUE)

spelling_variant:
	uv run --no-sync python -m alfs.update.refinement.spelling_variant \
		--senses-db $(SENSES_DB) \
		--cc-tasks-dir $(CC_TASKS_DIR)

morph_redirect:
	uv run --no-sync python -m alfs.update.refinement.morph_redirect \
		--senses-db $(SENSES_DB) \
		--queue-dir $(CLERK_QUEUE) \
		--model $(SENSE_UPDATE_MODEL)

undo_morph:
	uv run --no-sync python -m alfs.update.refinement.undo_morph \
		--senses-db $(SENSES_DB) \
		--queue-dir $(CLERK_QUEUE) \
		--model $(SENSE_UPDATE_MODEL)

trim_senses:
	uv run --no-sync python -m alfs.update.refinement.trim_sense \
		--senses-db $(SENSES_DB) \
		--labeled-db $(LABELED_DB) \
		--docs $(DOCS) \
		--queue-dir $(CLERK_QUEUE) \
		--n 50 \
		--model $(SENSE_UPDATE_MODEL)

delete_entry:
	uv run --no-sync python -m alfs.update.refinement.delete_entry \
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
		--queue-dir $(CLERK_QUEUE)

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
		--queue-dir $(CLERK_QUEUE)

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

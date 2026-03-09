.PHONY: etl seg update relabel label_new dedupe postag cleanup rewrite retag prune morph_redirect undo_morph trim_senses validate compile viewer backup backup-gdrive conductor clerk clerk-watch cc_apply cc-clean install_precommit_hooks dev test mypy cleandata

SENSES_DB          ?= ../alfs_data/senses.db
LABELED_DB         ?= ../alfs_data/labeled.db
CLERK_QUEUE        ?= ../clerk_queue
DOCS               ?= ../text_data/latest/docs.parquet
SENSES_REPO        ?= ../alfs_senses
SEG_DATA_DIR       ?= ../seg_data/latest/by_prefix
GDRIVE_REMOTE      ?= gdrive
GDRIVE_DEST        ?= alfs_backup
NWORDS             ?= 5
SENSE_UPDATE_MODEL ?= qwen2.5:32b
LABEL_MODEL        ?= gemma2:9b
CC_TASKS_DIR       ?= ../cc_tasks

etl:
	bash scripts/etl.sh

seg:
	bash scripts/segment.sh

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
		--model $(LABEL_MODEL)

label_new:
	bash scripts/label_new.sh \
		--senses-db $(SENSES_DB) \
		--labeled-db $(LABELED_DB) \
		--docs $(DOCS) \
		--seg-data-dir $(SEG_DATA_DIR) \
		--nwords $(NWORDS) \
		--model $(LABEL_MODEL)

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
		--senses-db $(SENSES_DB)

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

clerk:
	uv run --no-sync python -m alfs.clerk.worker \
		--queue-dir $(CLERK_QUEUE) \
		--senses-db $(SENSES_DB) \
		--labeled-db $(LABELED_DB)

cc_apply:
	uv run --no-sync python -m alfs.cc.apply \
		--cc-tasks-dir $(CC_TASKS_DIR) \
		--senses-db $(SENSES_DB) \
		--queue-dir $(CLERK_QUEUE)

cc-clean:
	rm -f $(CC_TASKS_DIR)/pending/*.json $(CC_TASKS_DIR)/done/*.json

clerk-watch:
	uv run --no-sync python -m alfs.clerk.worker \
		--queue-dir $(CLERK_QUEUE) \
		--senses-db $(SENSES_DB) \
		--labeled-db $(LABELED_DB) \
		--watch

validate:
	uv run --no-sync python -m alfs.qc.validate_labels \
		--labeled-db $(LABELED_DB) --docs $(DOCS)

compile:
	bash scripts/compile.sh

viewer:
	bash scripts/viewer.sh

backup:
	uv run --no-sync python -m alfs.backup \
		--senses-db $(SENSES_DB) --senses-repo $(SENSES_REPO)

backup-gdrive:
	rclone sync ../text_data $(GDRIVE_REMOTE):$(GDRIVE_DEST)/text_data \
		--exclude "cache/**" \
		--exclude "latest" \
		--progress
	rclone sync ../alfs_data $(GDRIVE_REMOTE):$(GDRIVE_DEST)/alfs_data \
		--exclude "*.db-wal" \
		--exclude "*.db-shm" \
		--progress
	rclone sync ../seg_data $(GDRIVE_REMOTE):$(GDRIVE_DEST)/seg_data \
		--exclude "latest" \
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

cleandata:
	@echo "This will delete: ../alfs_data  ../seg_data  ../update_data  ../viewer_data"
	@echo "text_data (ETL output) will NOT be touched."
	@read -p "Continue? [y/N] " ans && [ "$$ans" = "y" ] || { echo "Aborted."; exit 1; }
	rm -rf ../alfs_data ../seg_data ../update_data ../viewer_data

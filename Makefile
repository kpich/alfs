.PHONY: etl seg update relabel dedupe postag cleanup rewrite validate compile viewer backup conductor queenant install_precommit_hooks dev test mypy cleandata

SENSES_DB  ?= ../alfs_data/senses.db
LABELED_DB ?= ../alfs_data/labeled.db
CHANGES_DB ?= ../alfs_data/changes.db
DOCS       ?= ../text_data/latest/docs.parquet
SENSES_REPO ?= ../alfs_senses
NWORDS     ?= 5

etl:
	bash scripts/etl.sh

seg:
	bash scripts/segment.sh

update:
	bash scripts/update.sh

relabel:
	bash scripts/relabel.sh --nwords $(NWORDS)

dedupe:
	uv run --no-sync python -m alfs.update.refinement.dedupe \
		--senses-db $(SENSES_DB)

postag:
	uv run --no-sync python -m alfs.update.refinement.postag \
		--senses-db $(SENSES_DB) \
		--labeled-db $(LABELED_DB) \
		--docs $(DOCS)

cleanup:
	uv run --no-sync python -m alfs.update.refinement.cleanup \
		--senses-db $(SENSES_DB)

rewrite:
	uv run --no-sync python -m alfs.update.refinement.rewrite \
		--senses-db $(SENSES_DB) \
		--changes-db $(CHANGES_DB)

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

conductor:
	uv run --no-sync python -m alfs.anthill

queenant:
	uv run --no-sync python -m alfs.queenant \
		--senses-db $(SENSES_DB) \
		--changes-db $(CHANGES_DB)

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

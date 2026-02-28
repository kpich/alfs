.PHONY: etl seg update relabel dedupe postag compile viewer install_precommit_hooks dev test mypy cleandata

etl:
	bash scripts/etl.sh

seg:
	bash scripts/segment.sh

update:
	bash scripts/update.sh

relabel:
	bash scripts/relabel.sh

dedupe:
	uv run --no-sync python -m alfs.update.refinement.dedupe \
		--alfs ../alfs_data/alfs.json --output ../alfs_data/alfs.json

postag:
	uv run --no-sync python -m alfs.update.refinement.postag \
		--alfs ../alfs_data/alfs.json \
		--labeled ../alfs_data/labeled.parquet \
		--docs ../text_data/latest/docs.parquet \
		--output ../alfs_data/alfs.json

compile:
	bash scripts/compile.sh

viewer:
	bash scripts/viewer.sh

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

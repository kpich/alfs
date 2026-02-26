.PHONY: etl install_precommit_hooks

etl:
	bash scripts/etl.sh

install_precommit_hooks:
	uv sync --group dev
	uv run pre-commit install

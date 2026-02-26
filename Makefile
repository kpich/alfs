.PHONY: etl install_precommit_hooks dev test mypy

etl:
	bash scripts/etl.sh

install_precommit_hooks:
	uv sync --group dev
	uv run pre-commit install

dev:
	uv sync --group dev

test:
	uv run pytest

mypy:
	uv run mypy src/

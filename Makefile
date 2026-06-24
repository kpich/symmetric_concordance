.PHONY: dev test ruff mypy hooks

dev:
	uv sync

test:
	uv run pytest -q

ruff:
	uv run ruff check . && uv run ruff format --check .

mypy:
	uv run mypy

hooks:
	uv run pre-commit install

.PHONY: dev test ruff mypy hooks

# uv manages a project-local .venv automatically. `uv run` (and `uv sync`)
# create/sync it on demand and include the `dev` dependency group by default,
# so the targets below Just Work without any manual activation.

dev:
	uv sync

test:
	uv run pytest -q

ruff:
	uv run ruff check . && uv run ruff format --check .

mypy:
	uv run mypy

# Install the git pre-commit hook (requires a git repo). Hooks then run on commit.
hooks:
	uv run pre-commit install

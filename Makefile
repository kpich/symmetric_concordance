.PHONY: dev test ruff mypy hooks clean build publish-test publish

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

# --- Packaging / release -----------------------------------------------------
# Remove old build artifacts so a stale wheel never gets uploaded by mistake.
clean:
	rm -rf dist

# Build the sdist + wheel into dist/. Always start from a clean slate.
build: clean
	uv build

# Dry-run upload to TestPyPI (the practice site). Reads the token from the
# UV_PUBLISH_TOKEN env var, so it never lands in your shell history.
publish-test: build
	uv publish --publish-url https://test.pypi.org/legacy/

# Upload to the real PyPI. Same token mechanism. This is permanent per version.
publish: build
	uv publish

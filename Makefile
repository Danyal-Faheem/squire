# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Squire Makefile.
#
# Self-documenting targets: each target carries a `## description` that the
# `help` target prints as a menu. Code quality is handled by ruff (formatter +
# linter) and mypy (static types).

SRC_DIRS = squire tests
.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------
test: test-lint test-types test-unit ## Run the full test suite (lint + types + unit)

ci: test ## Alias for the checks CI runs (lint + types + unit)

test-lint: ## Check formatting and lint rules (does not modify files)
	ruff format --check $(SRC_DIRS)
	ruff check $(SRC_DIRS)

test-types: ## Run static type checks
	mypy $(SRC_DIRS)

test-unit: ## Run unit tests
	pytest

format: ## Format code and auto-fix lint issues
	ruff format $(SRC_DIRS)
	ruff check --fix $(SRC_DIRS)

# ---------------------------------------------------------------------------
# Local environment
# ---------------------------------------------------------------------------
install: ## Install the package with dev dependencies
	pip install -e ".[dev]"

up: ## Start the local stack (Postgres + services) in the background
	docker compose up -d

down: ## Stop the local stack
	docker compose down

logs: ## Tail logs from the local stack
	docker compose logs -f

migrate: ## Apply database migrations
	alembic upgrade head

ingest: ## Run the ingestion backfill (all activated connectors)
	python -m squire.ingestion.run_ingest --all

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help: ## Print this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

.PHONY: test ci test-lint test-types test-unit format install up down logs migrate ingest help

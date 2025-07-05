.DEFAULT_GOAL := help

PACKAGE_NAME := omnivault
TEST_DIR := tests
SOURCES := $(PACKAGE_NAME) $(TEST_DIR)

.PHONY: .uv
.uv:
	@uv -V || echo 'Please install uv: https://docs.astral.sh/uv/getting-started/installation/'

.PHONY: install
install: .uv
	uv sync --frozen --all-groups --all-extras
	uv run pre-commit install
	uv run pre-commit install --hook-type commit-msg

.PHONY: format
format: .uv
	uv run ruff check --fix --exit-zero $(SOURCES)
	uv run ruff format $(SOURCES)

.PHONY: lint
lint: .uv
	uv run ruff check $(SOURCES)
	uv run ruff format --check $(SOURCES)


.PHONY: security
security: .uv
	uv run bandit -r $(PACKAGE_NAME) -ll

.PHONY: typecheck
typecheck: .uv
	uv run mypy $(SOURCES)
	uv run pyright $(SOURCES)
	# @echo "Running ty (experimental)..."
	# uv run ty check $(SOURCES) || echo "ty check failed (expected for pre-release)"

.PHONY: test
test: .uv
	uv run pytest $(TEST_DIR)

.PHONY: coverage
coverage: .uv
	uv run coverage run -m pytest $(TEST_DIR)
	uv run coverage run -m pytest
	uv run coverage xml -o coverage.xml
	uv run coverage report -m --fail-under=95

.PHONY: docs
docs:
	cd docs && make html

.PHONY: ci
ci: format lint typecheck test coverage

.PHONY: lock
lock: .uv
	uv lock --upgrade

.PHONY: clean
clean:
	@./scripts/clean.sh

.PHONY: help
help:
	@echo "Development Commands:"
	@echo "  install             Install all dependencies (all groups + extras)"
	@echo "  install-lint        Install only linting dependencies"
	@echo "  install-test        Install only testing dependencies"
	@echo "  install-dev         Install dev group (includes lint, type, test, docs)"
	@echo "  format              Format code with ruff"
	@echo "  lint                Lint code with ruff (includes format check)"
	@echo "  security            Run security checks with bandit"
	@echo "  typecheck           Run type checking with mypy, pyright"
	@echo "  test                Run tests with pytest"
	@echo "  coverage            Run tests with coverage reporting"
	@echo "  docs                Build documentation"
	@echo "  ci                  Run full CI pipeline (lint, typecheck, test, coverage)"
	@echo ""
	@echo "Utility Commands:"
	@echo "  lock                Update lock files"
	@echo "  clean               Clean build artifacts and cache files"
	@echo "  help                Show this help message"
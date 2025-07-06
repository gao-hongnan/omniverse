.DEFAULT_GOAL := help # NOTE: if you run `make` without any arguments, it will run the help command

PACKAGE_NAME := omnivault
DOCS_DIR := omniverse
TEST_DIR := tests
SOURCES := $(PACKAGE_NAME) $(TEST_DIR)

.PHONY: .uv
.uv:
	@uv -V || echo 'Please install uv: https://docs.astral.sh/uv/getting-started/installation/'

.PHONY: install
install: .uv
	uv sync --frozen --all-extras --all-packages --all-groups
	uv run pre-commit install
	uv run pre-commit install --hook-type commit-msg

.PHONY: lock
lock: .uv
	uv lock --upgrade

.PHONY: sync
sync: .uv
	uv sync --all-extras --all-packages --all-groups

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
	uv run coverage html -d htmlcov
	uv run coverage xml -o coverage.xml
	uv run coverage report -m --fail-under=95

.PHONY: docs
docs: .uv
	cd $(DOCS_DIR) && uv run jupyter book start .

.PHONY: ci
ci: lint security typecheck test coverage

.PHONY: clean
clean:
	@./scripts/clean.sh

.PHONY: help
help:
	@echo "Development Commands:"
	@echo "  install             Install all dependencies (all groups + extras)"
	@echo "  lock                Update and regenerate lock file"
	@echo "  sync                Sync dependencies (without --frozen)"
	@echo "  format              Format code with ruff"
	@echo "  lint                Lint code with ruff (includes format check)"
	@echo "  security            Run security checks with bandit"
	@echo "  typecheck           Run type checking with mypy, pyright"
	@echo "  test                Run tests with pytest"
	@echo "  coverage            Run tests with coverage reporting (95% minimum)"
	@echo "  docs                Build Jupyter Book documentation"
	@echo "  ci                  Run full CI pipeline (lint, security, typecheck, test, coverage)"
	@echo ""
	@echo "Utility Commands:"
	@echo "  clean               Clean build artifacts and cache files"
	@echo "  help                Show this help message"
# Reference:
# https://github.com/langchain-ai/langchain/blob/master/libs/langchain/Makefile

.PHONY: lint test

# Default target executed when no arguments are given to make.
all: help

##########################
# LINTING AND FORMATTING #
##########################

# Lint target
lint:
	@echo "Running lint..."
	@./scripts/devops/continuous-integration/ci_lint_ruff.sh

# Test target
unit_test:
	@echo "Running unit tests..."
	@./scripts/devops/continuous-integration/ci_unit_test_pytest.sh

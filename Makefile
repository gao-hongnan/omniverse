# Reference:
# https://github.com/langchain-ai/langchain/blob/master/libs/langchain/Makefile

.PHONY: all security-check lint format isort black type-check unit-test

# Default target executed when no arguments are given to make.
all: security-check lint format type-check unit-test

######################
# SECURITY CHECKS     #
######################
security-check:
	@echo "Running security checks..."
	@bash -c 'set -o pipefail; \
	curl -sSL https://raw.githubusercontent.com/gao-hongnan/omniverse/continuous-integration/scripts/devops/continuous-integration/security_bandit.sh | \
	bash'

######################
# LINTING            #
######################
lint:
	@echo "Running lint..."
	@bash -c 'set -o pipefail; \
	curl -sSL https://raw.githubusercontent.com/gao-hongnan/omniverse/continuous-integration/scripts/devops/continuous-integration/lint_ruff.sh | \
	bash'

######################
# FORMATTING         #
######################
format: isort black

isort:
	@echo "Running Isort..."
	@bash -c 'set -o pipefail; \
	export CUSTOM_FLAGS=--filter-files; \
	curl -sSL https://raw.githubusercontent.com/gao-hongnan/omniverse/continuous-integration/scripts/devops/continuous-integration/format_isort.sh | \
	bash'

black:
	@echo "Running Black..."
	@bash -c 'set -o pipefail; \
	curl -sSL https://raw.githubusercontent.com/gao-hongnan/omniverse/continuous-integration/scripts/devops/continuous-integration/format_black.sh | \
	bash'

######################
# TYPE CHECKING      #
######################
type-check:
	@echo "Running MyPy type checks..."
	@bash -c 'set -o pipefail; \
	export CUSTOM_PACKAGES="omnivault" && \
	export CUSTOM_FLAGS="--python-version=3.9 --color-output --no-pretty" && \
	curl -sSL https://raw.githubusercontent.com/gao-hongnan/omniverse/continuous-integration/scripts/devops/continuous-integration/type_mypy.sh | \
	bash'

######################
# UNIT TESTS         #
######################
unit-test:
	@echo "Running unit tests..."
	@bash -c 'set -o pipefail; \
	curl -sSL https://raw.githubusercontent.com/gao-hongnan/omniverse/continuous-integration/scripts/devops/continuous-integration/test_unit_pytest.sh | \
	bash'

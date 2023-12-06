#!/usr/bin/env sh

set -e # exit on first error
# set -x # debug mode so expect verbose output

FLAGS=(
    --config=.ruff.toml
    --no-fix
    --no-show-source
    # Add more flags here if necessary
)
PACKAGES="omnivault" # _tmp_types.py"

SCRIPT_URL="https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/utils.sh"
UTILS_SCRIPT=$(curl -s "$SCRIPT_URL")

# Check if the fetch was successful
if [ $? -eq 0 ] && [ -n "$UTILS_SCRIPT" ]; then
    source /dev/stdin <<<"$UTILS_SCRIPT"
    # check_bash_version # TODO: macOS old bash version does not support readarray.
    logger "INFO" "Successfully fetched and sourced '$SCRIPT_URL'."
    logger "WARN" "🌈🌈🌈 Using custom logger for rich-like logging. Please put on your sunglasses 😎😎😎"
    logger "INFO" "Current working directory: $(pwd)"
    logger "INFO" "Current user: $(whoami)"
else
    echo "ERROR: Failed to fetch the '$SCRIPT_NAME' script from '$SCRIPT_URL'. Check your internet connection or access permissions."
    exit 1
fi

empty_line

usage_ruff() {
    logger "INFO" "Runs ruff with the specified options."
    logger "INFO" "Usage: lint_ruff [--<option>=<value>] ..."
    empty_line

    logger "INFO" "For more details, see link(s) below:"
    logger "LINK" "https://docs.astral.sh/ruff/configuration"
    logger "LINK" "https://docs.astral.sh/ruff/rules/"
    logger "CODE" "$ ruff --help"
    empty_line

    logger "INFO" "Example:"
    logger "BLOCK" \
    "$ ci_ruff_check \\
    --config=.ruff.toml \\
    --no-fix \\
    --no-show-source \\
    <PACKAGE-1> <PACKAGE-2> ..."
}

if check_for_help "$@"; then
    usage_ruff
    exit 0
fi

check_if_installed "ruff"

logger "INFO" "Running ruff linting with flags: ${FLAGS[*]} and packages: $PACKAGES. Please change them if necessary."
logger "INFO" "ruff version: $(ruff --version)"

# Linting with ruff
ruff check "${FLAGS[@]}" $PACKAGES

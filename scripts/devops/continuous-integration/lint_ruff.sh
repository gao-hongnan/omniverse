#!/usr/bin/env sh

########################## USER CONFIG #########################################
# Default FLAGS and PACKAGES are defined below. These settings can be overridden
# by environment variables to avoid allowing arbitrary flags within the script.
# This approach ensures controlled configuration while maintaining script security
# and integrity, unlike direct command-line overrides.
DEFAULT_FLAGS=(
    check
    --no-fix
    --no-show-source
    # Add more default flags here if necessary
)
DEFAULT_PACKAGES=(
    "omnivault"
    "tests"
    # Add more default packages here if necessary
)

# Use environment variables if set, otherwise use the defaults
FLAGS=(${CUSTOM_FLAGS:-"${DEFAULT_FLAGS[@]}"})
PACKAGES=(${CUSTOM_PACKAGES:-"${DEFAULT_PACKAGES[@]}"})

################################################################################

readonly SCRIPT_URL="https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/utils.sh"
readonly ALLOWED_CONFIG_FILES=(".ruff.toml" "ruff.toml" "pyproject.toml")
readonly TOOL="ruff"

UTILS_SCRIPT=$(curl -s "$SCRIPT_URL")

fetch_and_source_utils_script() {
    # Check if the fetch was successful
    if [ $? -eq 0 ] && [ -n "$UTILS_SCRIPT" ]; then
        source /dev/stdin <<<"$UTILS_SCRIPT"
        # check_bash_version # TODO: macOS old bash version does not support readarray.
        logger "INFO" "Successfully fetched and sourced '$SCRIPT_URL'."
        logger "WARN" "🌈🌈🌈 Using custom logger for rich-like logging."
        logger "INFO" "Current working directory: $(pwd)"
        logger "INFO" "Current user: $(whoami)"
    else
        echo "ERROR: Failed to fetch the script from '$SCRIPT_URL'. Check your internet connection or access permissions."
        exit 1
    fi
}

log_env_override() {
    if [ -n "$CUSTOM_FLAGS" ]; then
        logger "WARN" "Custom flags provided via environment variable: $CUSTOM_FLAGS"
    else
        logger "INFO" "Using default flags: ${DEFAULT_FLAGS[*]}"
    fi

    if [ -n "$CUSTOM_PACKAGES" ]; then
        logger "WARN" "Custom packages provided via environment variable: $CUSTOM_PACKAGES"
    else
        logger "INFO" "Using default packages: ${DEFAULT_PACKAGES[*]}"
    fi
}

show_usage() {
    logger "INFO" "Runs ${TOOL} with the specified options."
    logger "INFO" "Usage: ${TOOL} [--<option>=<value>] ..."
    empty_line

    logger "INFO" "For more details, see link(s) below:"
    logger "LINK" "https://docs.astral.sh/${TOOL}/"
    logger "LINK" "https://docs.astral.sh/${TOOL}/configuration"
    logger "LINK" "https://docs.astral.sh/${TOOL}/rules/"
    logger "CODE" "$ ${TOOL} --help"
    logger "BLOCK" \
    "$ ${TOOL} --check \\
    --config=.${TOOL}.toml \\
    --no-fix \\
    --no-show-source \\
    <PACKAGE-1> <PACKAGE-2> ..."
    empty_line

    logger "INFO" "Usage: $(basename $0) [OPTIONS]"
    empty_line

    logger "INFO" "Options:"
    logger "CODE" "  --debug    Enable debug mode"
    logger "CODE" "  --strict   Exit script on first error"
    logger "CODE" "  -h, --help Show this help message"
    empty_line

    logger "INFO" "Example:"
    logger "CODE" "  $(basename $0) --debug --strict"
}

main() {
    fetch_and_source_utils_script # need to call first because it uses logger

    log_env_override

    # Parse command-line arguments
    while [ "$#" -gt 0 ]; do
        case $1 in
            --debug)
                debug_mode=true
                ;;
            --strict)
                strict_mode=true
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                logger "ERROR" "Unrecognized argument: $1"
                exit 1
                ;;
        esac
        shift
    done

    if [ "$debug_mode" = true ]; then
        set -x # Debug mode: print each command
    fi

    if [ "$strict_mode" = true ]; then
        set -e # Exit on first error
    fi

    check_if_installed "${TOOL}"
    check_config_files "${ALLOWED_CONFIG_FILES[@]}"

    logger "INFO" "Running ${TOOL} linting with flags: ${FLAGS[*]} and packages: ${PACKAGES[*]}. Please change them if necessary."
    logger "INFO" "${TOOL} version: $(${TOOL} --version)"

    local cmd="${TOOL} ${FLAGS[@]} ${PACKAGES[@]}"
    logger "INFO" "The ${TOOL} command to be executed:"
    logger "CODE" "$cmd"

    $cmd

    local status=$?

    if [ "$status" -eq 0 ]; then
        logger "INFO" "🎉🎉🎉 ${TOOL} linting passed."
    else
        logger "ERROR" "💥💥💥 ${TOOL} linting failed."
    fi

    exit $status
}

main "$@"

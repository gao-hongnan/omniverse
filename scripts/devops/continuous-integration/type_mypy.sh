#!/usr/bin/env sh

########################## USER CONFIG #########################################
# FLAGS and PACKAGES are defined here as a config for users to change instead
# of allowing arbitrary flags in the script. So in the argparse below, we
# do not allow FLAGS and PACKAGES to override the cli.
FLAGS=(
    --python-version=3.9
    --color-output
    --pretty
    # Add more flags here if necessary
)
PACKAGES=(
    "omnivault"
    # Add more packages here if necessary
)

################################################################################

readonly SCRIPT_URL="https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/utils.sh"
readonly ALLOWED_CONFIG_FILES=(
    "./mypy.ini"
    "./.mypy.ini"
    "./pyproject.toml"
    "./setup.cfg"
    "$XDG_CONFIG_HOME/mypy/config"
    "~/.config/mypy/config"
    "~/.mypy.ini"
)
readonly TOOL="mypy"

UTILS_SCRIPT=$(curl -s "$SCRIPT_URL")

fetch_and_source_utils_script() {
    # Check if the fetch was successful
    if [ $? -eq 0 ] && [ -n "$UTILS_SCRIPT" ]; then
        source /dev/stdin <<<"$UTILS_SCRIPT"
        # check_bash_version # TODO: macOS old bash version does not support readarray.
        logger "INFO" "Successfully fetched and sourced '$SCRIPT_URL'."
        logger "WARN" "🌈🌈🌈 Using custom logger for rich-like logging. Please put on your sunglasses 😎😎😎"
        logger "INFO" "Current working directory: $(pwd)"
        logger "INFO" "Current user: $(whoami)"
    else
        echo "ERROR: Failed to fetch the script from '$SCRIPT_URL'. Check your internet connection or access permissions."
        exit 1
    fi
}

show_usage() {
    logger "INFO" "Runs ${TOOL} with the specified options."
    logger "INFO" "Usage: ${TOOL} [--<option>=<value>] ..."
    empty_line

    logger "INFO" "For more details, see link(s) below:"
    logger "LINK" "https://${TOOL}.readthedocs.io/en/stable/index.html"
    logger "LINK" "https://${TOOL}.readthedocs.io/en/stable/config_file.html"
    logger "CODE" "$ ${TOOL} --help"
    logger "BLOCK" \
    "$ ${TOOL} --config=.${TOOL}.toml \\
    --python-version=3.9 \\
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
}

main "$@"

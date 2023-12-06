#!/usr/bin/env sh

set -e
set -x

echo "Current directory: $PWD"

PACKAGES="omnivault _tmp_types.py"
FLAGS=(
    --config=.ruff.toml
    --no-fix
    --no-show-source
    # Add more flags here if necessary
)

# Linting with ruff
ruff check "${FLAGS[@]}" $PACKAGES
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Project Overview

Omniverse is a comprehensive machine learning, deep learning, and software
engineering repository featuring:

-   Machine learning implementations (clustering, linear models, naive bayes,
    etc.)
-   Deep learning transformer architectures (GPT-style decoder-only models)
-   Data structures and algorithms implementations
-   Linear algebra fundamentals
-   Distributed training capabilities
-   Jupyter Book documentation

## Core Architecture

### Package Structure

-   `omnivault/` - Main package containing core implementations
    -   `transformer/` - Transformer model implementations with GPT-like
        architecture
    -   `machine_learning/` - Classic ML algorithms (clustering, linear models,
        generative models)
    -   `modules/` - Neural network components (activation, loss, pooling, LoRA)
    -   `distributed/` - Distributed training utilities
    -   `dsa/` - Data structures and algorithms
    -   `linear_algebra/` - Vector and matrix operations
    -   `utils/` - Utilities for config management, reproducibility,
        visualization

### Key Components

-   **Transformer Implementation**: Located in `omnivault/transformer/`,
    includes decoder-only GPT-style models with projects for adder tasks,
    TinyShakespeare, and SimpleBooks-92
-   **Training Framework**: Comprehensive trainer with support for mixed
    precision, gradient scaling, gradient accumulation, and distributed training
-   **Configuration Management**: Uses Hydra for configuration with YAML files
-   **Reproducibility**: Includes utilities for deterministic training and RNG
    state management

## Development Commands

### Environment Setup

```bash
# Install all dependency groups + extras (uses uv; `dev` is a PEP 735
# dependency-group, not an extra, so `pip install -e ".[dev]"` does NOT work)
make install

# Sync only, without the pre-commit install step
uv sync --all-extras --all-packages --all-groups

# Install serving dependencies (the only extra)
pip install -e ".[serving]"
```

### Common Development Tasks

```bash
# Run all checks (lint, security, type-check, tests)
make ci

# Individual tasks
make security         # Run bandit security checks
make lint             # Run ruff check + ruff format --check
make format           # Apply ruff fixes and formatting
make typecheck        # Run mypy AND pyright
make test             # Run pytest
make coverage         # Run pytest with coverage

# Manual commands
pytest tests/omnivault/unit --verbose    # Run unit tests
ruff check omnivault/                    # Lint specific package
ruff format omnivault/                   # Format specific package
mypy omnivault/ --python-version=3.14   # Type check specific package
```

### Testing

-   Unit tests: `tests/omnivault/unit/`
-   Run specific test: `pytest tests/omnivault/unit/path/to/test.py -v`
-   Coverage reports via `make coverage`

### Transformer Training

```bash
# Example training commands for transformer projects
cd omnivault/transformer/projects/adder/
python main.py config.yaml data.train_loader.batch_size=256 trainer.max_epochs=10

# Distributed training
python main_distributed.py config_distributed.yaml
```

### Building and Deployment

```bash
# Build package
python -m build

# Docker builds
docker build --file docker/nvidia/omniverse-nvidia.Dockerfile --tag omniverse-nvidia .
docker build --file docker/documentation/jupyterbook.Dockerfile --tag omniverse-docs .
```

## Configuration and Tools

### Code Quality

-   **Formatter**: Ruff (120 char line length)
-   **Linter**: Ruff
-   **Type Checkers**: MyPy and Pyright (Python 3.14 target)
-   **Security**: Bandit
-   **Testing**: pytest with coverage support

### Key Configuration Files

-   `pyproject.toml` - Project configuration, dependencies, tool settings
-   `config.yaml` - Main project configuration
-   Transformer configs in `omnivault/transformer/projects/*/config.yaml`

### CI/CD Scripts

Located in `scripts/devops/continuous-integration/`:

-   All scripts fetch from remote URLs for consistency
-   Support custom flags via environment variables (e.g., `CUSTOM_FLAGS`,
    `CUSTOM_PACKAGES`)

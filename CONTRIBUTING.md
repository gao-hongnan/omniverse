# Contributing to Omniverse

Thank you for your interest in contributing to Omniverse! This guide provides
comprehensive instructions for setting up your development environment and
contributing to the project.

## Table of Contents

- [Contributing to Omniverse](#contributing-to-omniverse)
    - [Table of Contents](#table-of-contents)
    - [Code of Conduct](#code-of-conduct)
    - [Development Setup](#development-setup)
        - [Prerequisites](#prerequisites)
        - [Environment Setup](#environment-setup)
        - [Dependencies Installation](#dependencies-installation)
    - [Development Workflow](#development-workflow)
        - [Branch Strategy](#branch-strategy)
        - [Commit Conventions](#commit-conventions)
        - [Pre-commit Hooks](#pre-commit-hooks)
    - [Code Standards](#code-standards)
        - [Code Style](#code-style)
        - [Type Annotations](#type-annotations)
        - [Docstrings](#docstrings)
        - [Import Organization](#import-organization)
    - [Testing Guidelines](#testing-guidelines)
        - [Writing Tests](#writing-tests)
        - [Running Tests](#running-tests)
        - [Coverage Requirements](#coverage-requirements)
    - [Making Contributions](#making-contributions)
        - [Reporting Issues](#reporting-issues)
        - [Submitting Pull Requests](#submitting-pull-requests)
        - [Code Review Process](#code-review-process)
    - [CI/CD Pipeline](#cicd-pipeline)
    - [Documentation](#documentation)
        - [Building Documentation](#building-documentation)
        - [Documentation Standards](#documentation-standards)
        - [Updating Documentation](#updating-documentation)
    - [Project Structure](#project-structure)
    - [Common Development Tasks](#common-development-tasks)
        - [Makefile Commands Reference](#makefile-commands-reference)
        - [Dependency Management with uv](#dependency-management-with-uv)
    - [Performance Considerations](#performance-considerations)
        - [Machine Learning Code](#machine-learning-code)
        - [Profiling Tools](#profiling-tools)
    - [Debugging Tips](#debugging-tips)
        - [General Debugging](#general-debugging)
        - [PyTorch Debugging](#pytorch-debugging)
        - [GPU/CUDA Debugging](#gpucuda-debugging)
    - [Release Process](#release-process)
        - [Version Management](#version-management)
        - [Release Checklist](#release-checklist)
    - [Getting Help](#getting-help)
        - [Resources](#resources)
        - [Communication Channels](#communication-channels)
        - [Tips for Getting Help](#tips-for-getting-help)

## Code of Conduct

This project is released with a [Contributor Code of Conduct](CONDUCT.md). By
participating in this project, you agree to abide by its terms.

## Development Setup

### Prerequisites

1. **Python 3.14+**: This project requires Python 3.14 or higher.

    ```bash
    python --version  # Should output Python 3.14.x or higher
    ```

2. **uv**: This project uses [uv](https://docs.astral.sh/uv/) for dependency
   management.

    ```bash
    # Install uv (if not already installed)
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

3. **Git**: Ensure you have Git installed and configured.

    ```bash
    git --version
    git config --global user.name "Your Name"
    git config --global user.email "your.email@example.com"
    ```

### Environment Setup

1. **Fork and Clone the Repository**

    ```bash
    # Fork the repository on GitHub, then clone your fork
    git clone https://github.com/YOUR_USERNAME/omniverse.git
    cd omniverse

    # Add upstream remote
    git remote add upstream https://github.com/gao-hongnan/omniverse.git
    ```

2. **Create a Development Environment**

    The project uses `uv` for dependency management, which handles virtual
    environments automatically:

    ```bash
    # Install all dependencies (including dev, test, lint, type, and docs groups)
    make install
    ```

    This command will:

    - Create a virtual environment using `uv`
    - Install all project dependencies and development tools
    - Set up pre-commit hooks for code quality checks
    - Install commit message hooks for conventional commits

### Dependencies Installation

The project uses a modular dependency structure defined in `pyproject.toml`:

-   **Core dependencies**: Machine learning and deep learning libraries
-   **Optional dependencies**: Serving components (`fastapi`, `uvicorn`, etc.)
-   **Development groups**:
    -   `lint`: Code formatting and linting tools (`ruff`, `bandit`)
    -   `type`: Type checking tools (`mypy`, `pyright`, type stubs)
    -   `test`: Testing framework (`pytest`, `coverage`)
    -   `docs`: Documentation tools (`jupyter-book`, Sphinx extensions)

To sync dependencies after making changes:

```bash
make sync  # Sync without --frozen flag for development
```

## Development Workflow

### Branch Strategy

1. **Main Branch**: `main` is the stable branch
2. **Feature Branches**: Create feature branches from `main`

    ```bash
    # Create a new feature branch
    git checkout main
    git pull upstream main
    git checkout -b feature/your-feature-name
    ```

3. **Branch Naming Conventions**:
    - `feature/` - New features
    - `fix/` - Bug fixes
    - `docs/` - Documentation updates
    - `refactor/` - Code refactoring
    - `test/` - Test additions or modifications
    - `chore/` - Maintenance tasks

### Commit Conventions

This project uses [Conventional Commits](https://www.conventionalcommits.org/)
with Commitizen. The format is:

```bash
# Basic format
<type>: <subject>

# With optional scope
<type>(<scope>): <subject>

# Examples
git commit -m "feat: add attention visualization"
git commit -m "feat(transformer): add rotary position embeddings"
git commit -m "fix(trainer): resolve gradient accumulation bug"
git commit -m "docs(readme): update installation instructions"
```

**Commit Types**:

-   `feat`: New feature
-   `fix`: Bug fix
-   `docs`: Documentation changes
-   `style`: Code style changes (formatting, semicolons, etc.)
-   `refactor`: Code refactoring
-   `perf`: Performance improvements
-   `test`: Test additions or modifications
-   `build`: Build system changes
-   `ci`: CI/CD changes
-   `chore`: Maintenance tasks

**Note**: Commitizen enforces these conventions through pre-commit hooks. The
scope is optional but recommended for clarity.

### Pre-commit Hooks

Pre-commit hooks are automatically installed during setup. They run on every
commit to ensure code quality:

```bash
# Run the git hooks manually on all files (prek is a drop-in, faster
# reimplementation of pre-commit and reads the same config file)
uv run prek run --all-files

# Skip hooks temporarily (not recommended)
git commit -m "message" --no-verify
```

The hooks include:

-   Commitizen for commit message validation
-   Ruff for linting and formatting
-   Bandit for security checks
-   MyPy and Pyright for type checking

## Code Standards

### Code Style

The project uses **Ruff** for both linting and formatting with a 120-character
line limit:

```bash
# Format code
make format

# Check code style
make lint
```

Key style guidelines:

-   Line length: 120 characters maximum
-   Use descriptive variable names
-   Follow PEP 8 with Ruff's modern interpretations
-   Organize imports with Ruff's import sorting

### Type Annotations

All code must include comprehensive type annotations using Python 3.14+ syntax:

```python
# Python 3.14+ style - no need to import from typing for basic types
from typing import Any, TypeVar, Generic
from collections.abc import Iterator, Callable
import torch
from torch import nn

T = TypeVar('T')

def process_batch(
    inputs: torch.Tensor,
    model: nn.Module,
    temperature: float = 1.0,
    top_k: int | None = None
) -> tuple[torch.Tensor, dict[str, float]]:
    """Process a batch of inputs through the model."""
    ...

# Use union types with | operator
def parse_config(path: str | Path) -> dict[str, Any]:
    ...

# Use built-in types directly
def get_tokens(text: str) -> list[str]:
    ...

# Generic types
class DataLoader(Generic[T]):
    def __init__(self, data: list[T]) -> None:
        self.data = data
```

Run type checking:

```bash
# Run both mypy and pyright
make typecheck

# Run individually
uv run mypy omnivault/ --python-version=3.14
uv run pyright omnivault/
```

### Docstrings

Use **NumPy-style** docstrings for all public functions, classes, and modules:

```python
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_accumulation_steps: int = 1
) -> dict[str, float]:
    """Train model for one epoch.

    Parameters
    ----------
    model : nn.Module
        The neural network model to train.
    dataloader : DataLoader
        DataLoader providing training batches.
    optimizer : Optimizer
        Optimizer for updating model parameters.
    criterion : nn.Module
        Loss function.
    device : torch.device
        Device to run training on (CPU/GPU).
    gradient_accumulation_steps : int, optional
        Number of steps to accumulate gradients, by default 1.

    Returns
    -------
    dict[str, float]
        Dictionary containing:
        - 'loss': Average loss for the epoch
        - 'accuracy': Training accuracy
        - 'learning_rate': Current learning rate

    Raises
    ------
    RuntimeError
        If CUDA out of memory occurs.

    Examples
    --------
    >>> metrics = train_epoch(model, loader, opt, loss_fn, device)
    >>> print(f"Loss: {metrics['loss']:.4f}")

    Notes
    -----
    This function implements gradient accumulation for handling
    larger effective batch sizes on limited GPU memory.
    """
    ...
```

### Import Organization

Imports should be organized in the following order (automatically handled by
Ruff):

1. Standard library imports
2. Third-party imports
3. Local application imports

```python
# Standard library
import os
from pathlib import Path
from collections.abc import Sequence

# Third-party
import torch
from torch import nn
import numpy as np

# Local
from omnivault.transformer.config import TransformerConfig
from omnivault.utils.reproducibility import seed_all
```

## Testing Guidelines

### Writing Tests

1. **Test Location**: Place tests in `tests/omnivault/unit/` mirroring the
   source structure
2. **Test Naming**: Use `test_` prefix for test files and functions
3. **Test Organization**: Group related tests in classes when appropriate
4. **Type Hints**: All test code must include type annotations
5. **Quality Standards**: Tests should follow the same code quality standards as
   production code

Example test structure:

```python
import pytest
import torch
from omnivault.transformer.modules.attention import MultiHeadAttention

class TestMultiHeadAttention:
    """Test cases for MultiHeadAttention module."""

    @pytest.fixture
    def attention_module(self) -> MultiHeadAttention:
        """Create a test attention module."""
        return MultiHeadAttention(d_model=512, n_heads=8)

    def test_forward_shape(self, attention_module: MultiHeadAttention) -> None:
        """Test output shape of attention module."""
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, 512)
        output = attention_module(x)
        assert output.shape == (batch_size, seq_len, 512)

    @pytest.mark.parametrize(
        "seq_len,expected_mask_shape",
        [(10, (10, 10)), (20, (20, 20))]
    )
    def test_attention_mask(
        self,
        attention_module: MultiHeadAttention,
        seq_len: int,
        expected_mask_shape: tuple[int, int]
    ) -> None:
        """Test attention with causal mask."""
        mask = attention_module.create_causal_mask(seq_len)
        assert mask.shape == expected_mask_shape
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
uv run pytest tests/omnivault/unit/transformer/test_attention.py -v

# Run with coverage
make coverage

# Run specific test with markers
uv run pytest -m "not slow" tests/

# Run with type checking
uv run pytest --typeguard-packages=omnivault
```

### Coverage Requirements

The project maintains a **95% code coverage minimum**:

```bash
# Generate coverage report
make coverage

# View HTML coverage report
open htmlcov/index.html
```

Coverage configuration in `pyproject.toml` excludes test files and specific
patterns.

## Making Contributions

### Reporting Issues

When reporting issues, please include:

1. **Environment Information**:

    - Operating system and version
    - Python version
    - PyTorch version
    - CUDA version (if applicable)

2. **Minimal Reproducible Example**:

    ```python
    # Code that reproduces the issue
    import omnivault
    # ... minimal code to reproduce
    ```

3. **Error Messages**: Complete error traceback

4. **Expected vs Actual Behavior**: Clear description of what should happen

### Submitting Pull Requests

1. **Prepare Your Changes**

    ```bash
    # Ensure your fork is up to date
    git checkout main
    git pull upstream main
    git push origin main

    # Create feature branch
    git checkout -b feature/your-feature
    ```

2. **Develop Your Feature**

    - Write code following the style guidelines
    - Add comprehensive tests with type hints
    - Update documentation as needed
    - Ensure all checks pass locally

3. **Run Quality Checks**

    ```bash
    # Run full CI pipeline locally
    make ci

    # Or run individual checks
    make format      # Format code
    make lint        # Lint check
    make security    # Security scan
    make typecheck   # Type checking
    make test        # Run tests
    make coverage    # Check coverage
    ```

4. **Commit and Push**

    ```bash
    # Make atomic commits with conventional commit messages
    git add -p  # Stage changes interactively
    git commit  # Commitizen will guide you

    # Push to your fork
    git push origin feature/your-feature
    ```

5. **Create Pull Request**
    - Open PR against `main` branch
    - Fill out the PR template completely
    - Link related issues
    - Ensure CI passes

### Code Review Process

1. **Reviewer Expectations**:

    - Be constructive and respectful
    - Focus on code quality and maintainability
    - Consider performance implications
    - Verify test coverage

2. **Author Responsibilities**:

    - Respond to feedback promptly
    - Keep PR focused and atomic
    - Update documentation if needed
    - Maintain clean commit history

3. **Review Checklist**:
    - [ ] Code follows style guidelines
    - [ ] Tests pass with adequate coverage
    - [ ] Type hints are comprehensive
    - [ ] Documentation is updated
    - [ ] Performance impact is considered
    - [ ] Security implications reviewed

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration. The pipeline runs:

1. **Code Quality Checks**:

    - Linting with Ruff
    - Security scanning with Bandit
    - Type checking with MyPy and Pyright

2. **Testing**:

    - Unit tests with pytest
    - Coverage reporting (95% minimum)
    - Matrix testing across Python versions

3. **Documentation Build**:
    - Jupyter Book compilation
    - Link checking

To run the full CI pipeline locally:

```bash
make ci
```

## Documentation

### Building Documentation

The project uses Jupyter Book for documentation:

```bash
# Build and serve documentation locally
make docs

# This runs:
# cd omniverse && uv run jupyter book start .
```

### Documentation Standards

1. **Module Documentation**: Every module should have a comprehensive docstring
2. **API Documentation**: All public APIs must be documented with NumPy-style
   docstrings
3. **Examples**: Include runnable examples in docstrings
4. **Jupyter Notebooks**: Place educational notebooks in `omniverse/` directory

### Updating Documentation

When adding new features:

1. Update relevant `.md` files in `omniverse/`
2. Add/update Jupyter notebooks for examples
3. Update the table of contents in `omniverse/_toc.yml`
4. Ensure all links work correctly

## Project Structure

```text
omniverse/
├── omnivault/              # Main package
│   ├── transformer/        # Transformer implementations
│   │   ├── projects/       # Example projects (adder, tinyshakespeare)
│   │   ├── modules/        # Core transformer components
│   │   └── config/         # Configuration management
│   ├── machine_learning/   # Classical ML algorithms
│   ├── modules/            # Neural network components
│   ├── distributed/        # Distributed training utilities
│   ├── dsa/                # Data structures and algorithms
│   ├── linear_algebra/     # Linear algebra operations
│   └── utils/              # Utilities (config, reproducibility, etc.)
├── tests/                  # Test suite
│   └── omnivault/
│       └── unit/           # Unit tests
├── omniverse/              # Documentation (Jupyter Book)
├── scripts/                # Utility scripts
│   └── devops/             # CI/CD scripts
├── docker/                 # Docker configurations
├── .github/                # GitHub Actions workflows
├── .pre-commit-config.yaml # Pre-commit hooks configuration
├── .markdownlint.json      # Markdown linting rules
├── pyrightconfig.json      # Pyright configuration
├── Makefile                # Development commands
├── pyproject.toml          # Project configuration
└── uv.lock                 # Locked dependencies
```

## Common Development Tasks

### Makefile Commands Reference

```bash
# Development Setup
make install    # Install all dependencies and hooks
make sync       # Sync dependencies without --frozen
make lock       # Update and regenerate lock file

# Code Quality
make format     # Auto-format code
make lint       # Check code style
make security   # Run security checks
make typecheck  # Type checking with mypy and pyright

# Testing
make test       # Run all tests
make coverage   # Run tests with coverage report

# Documentation
make docs       # Build and serve documentation

# Full Pipeline
make ci         # Run complete CI pipeline locally

# Utilities
make clean      # Clean build artifacts and caches
make help       # Show all available commands
```

### Dependency Management with uv

```bash
# Add a new dependency
uv add package-name

# Add to specific group
uv add --group test pytest-benchmark

# Update dependencies
uv lock --upgrade

# Show dependency tree
uv tree
```

## Performance Considerations

### Machine Learning Code

1. **Vectorization**: Prefer NumPy/PyTorch operations over loops
2. **Memory Management**: Use generators for large datasets
3. **GPU Utilization**: Profile CUDA kernels for bottlenecks
4. **Mixed Precision**: Use AMP for faster training

    ```python
    from torch.cuda.amp import autocast, GradScaler

    scaler = GradScaler()
    with autocast(dtype=torch.float16):
        output = model(inputs)
        loss = criterion(output, targets)
    ```

### Profiling Tools

```python
# Memory profiling
from omnivault.utils.memory_profiler import MemoryProfiler

with MemoryProfiler() as profiler:
    # Your code here
    pass

# Time profiling
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
) as prof:
    model(inputs)
```

## Debugging Tips

### General Debugging

1. **Enable Debug Logging**:

    ```python
    import logging
    logging.basicConfig(level=logging.DEBUG)

    # Or use rich for better formatting
    from rich.logging import RichHandler
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    ```

2. **Interactive Debugging**:

    ```python
    # Use breakpoint() for debugging
    breakpoint()  # Python 3.7+

    # Or use ipdb
    import ipdb; ipdb.set_trace()
    ```

### PyTorch Debugging

```python
# Enable anomaly detection for gradient debugging
torch.autograd.set_detect_anomaly(True)

# Check for NaN/Inf values
assert not torch.isnan(tensor).any(), "NaN detected"
assert not torch.isinf(tensor).any(), "Inf detected"

# Print tensor statistics
def tensor_stats(tensor: torch.Tensor, name: str) -> None:
    print(f"{name}: shape={tensor.shape}, "
          f"mean={tensor.mean():.4f}, "
          f"std={tensor.std():.4f}, "
          f"min={tensor.min():.4f}, "
          f"max={tensor.max():.4f}")
```

### GPU/CUDA Debugging

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Debug CUDA errors
export CUDA_LAUNCH_BLOCKING=1
```

```python
# Memory debugging
import torch.cuda

print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Clear cache
torch.cuda.empty_cache()

# Synchronize for accurate timing
torch.cuda.synchronize()
```

## Release Process

### Overview

The project uses an automated release process with:

- **uv**: For building and publishing packages
- **Commitizen**: For version management and changelog generation
- **GitHub Actions**: For automated releases
- **PyPI Trusted Publishing**: For secure package publishing

### Automated Releases (Recommended)

Releases are automatically triggered when conventional commits are merged to `main`:

- `feat:` commits trigger a **minor** version bump (0.1.0 → 0.2.0)
- `fix:` commits trigger a **patch** version bump (0.1.0 → 0.1.1)
- `feat!:` or `BREAKING CHANGE:` trigger a **major** version bump (0.1.0 → 1.0.0)

The automated workflow:
1. Analyzes commits and determines version bump
2. Updates version and generates changelog
3. Creates a version tag
4. Builds and publishes to TestPyPI
5. Runs installation tests
6. Publishes to PyPI
7. Creates a GitHub release

### Manual Version Management

For maintainers who need manual control:

```bash
# Check current version
uv version

# Bump version based on commits
uv run cz bump --changelog

# Specific version bump
uv run cz bump --increment MAJOR|MINOR|PATCH

# Create a prerelease
uv run cz bump --prerelease rc
```

### Test Releases

To test the release process without affecting the main package:

```bash
# Trigger test release workflow
gh workflow run publish-test.yaml \
  -f version_suffix=test1 \
  -f publish_testpypi=true
```

### Release Checklist

Before releasing:
- [ ] All CI checks passing
- [ ] Documentation updated
- [ ] Conventional commits used
- [ ] No uncommitted changes

For detailed release instructions, see [RELEASING.md](RELEASING.md).

## Getting Help

### Resources

-   **Documentation**: [Jupyter Book Documentation](omniverse/)
-   **Issues**: [GitHub Issues](https://github.com/gao-hongnan/omniverse/issues)
-   **Discussions**: Use GitHub Discussions for questions

### Communication Channels

1. **Bug Reports**: GitHub Issues with `bug` label
2. **Feature Requests**: GitHub Issues with `enhancement` label
3. **Questions**: GitHub Discussions
4. **Security Issues**: Email maintainers directly (see SECURITY.md)

### Tips for Getting Help

1. Search existing issues/discussions first
2. Provide minimal reproducible examples
3. Include system information
4. Be specific about expected vs actual behavior

---

Thank you for contributing to Omniverse! Your contributions help make this
project better for everyone.

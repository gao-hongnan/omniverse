"""This module contains utilities for seeding the random number generators to
ensure reproducibility.

Notes
-----
Reproducibility in deep learning ensures that experiments can be repeated with
identical results, critical for verifying research findings and deploying
reliable models. Distributed training introduces complexity because it involves
multiple computation units which may not synchronize their random states
perfectly. If training is paused and resumed, ensuring each unit starts with the
correct seed to reproduce the exact computational path becomes challenging. To
address this, one can find more sophisticated examples in libraries like
Composer, where the whole library's core is built around training deep neural
nets in any environment (distributed or not) with reproducibility in mind.

References
----------

- `PyTorch Reproducibility <https://pytorch.org/docs/stable/notes/randomness.html>`_
- `PyTorch deterministic algorithms <https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html>`_
- `CUBLAS reproducibility <https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility>`_
- `PyTorch Worker <https://pytorch.org/docs/stable/notes/randomness.html#dataloader>`_
- `Composer <https://github.com/mosaicml/composer/blob/dev/composer/utils/reproducibility.py>`_
"""

from __future__ import annotations

import os
import random
import warnings

import numpy as np
import torch
import torch.backends.cudnn

__all__ = [
    "seed_all",
    "seed_worker",
    "configure_deterministic_mode",
    "raise_error_if_seed_is_negative_or_outside_32_bit_unsigned_integer",
]

max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min


def raise_error_if_seed_is_negative_or_outside_32_bit_unsigned_integer(value: int) -> None:
    if not (min_seed_value <= value <= max_seed_value):
        raise ValueError(f"Seed must be within the range [{min_seed_value}, {max_seed_value}]")


def configure_deterministic_mode() -> None:
    r"""
    Activates deterministic mode in PyTorch and CUDA to ensure reproducible
    results at the cost of performance and potentially higher CUDA memory usage.
    It sets deterministic algorithms, disables cudnn benchmarking and enables,
    and sets the CUBLAS workspace configuration.

    References
    ----------
    - `PyTorch Reproducibility <https://pytorch.org/docs/stable/notes/randomness.html>`_
    - `PyTorch deterministic algorithms <https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html>`_
    - `CUBLAS reproducibility <https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility>`_
    """

    # fmt: off
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark        = False
    torch.backends.cudnn.deterministic    = True
    torch.backends.cudnn.enabled          = False

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # fmt: on
    warnings.warn(
        "Deterministic mode is activated. This will negatively impact performance and may cause increase in CUDA memory footprint.",
        category=UserWarning,
        stacklevel=2,
    )


def seed_all(
    seed: int = 1992,
    seed_torch: bool = True,
    set_torch_deterministic: bool = True,
) -> int:
    """
    Seeds all relevant random number generators to ensure reproducible
    outcomes. Optionally seeds PyTorch and activates deterministic
    behavior in PyTorch based on the flags provided.

    Parameters
    ----------
    seed : int, default=1992
        The seed number for reproducibility.
    seed_torch : bool, default=True
        If True, seeds PyTorch's RNGs.
    set_torch_deterministic : bool, default=True
        If True, activates deterministic mode in PyTorch.

    Returns
    -------
    seed : int
        The seed number used for reproducibility.
    """
    raise_error_if_seed_is_negative_or_outside_32_bit_unsigned_integer(seed)

    # fmt: off
    os.environ["PYTHONHASHSEED"] = str(seed)       # set PYTHONHASHSEED env var at fixed value
    np.random.seed(seed)                           # numpy pseudo-random generator
    random.seed(seed)                              # python's built-in pseudo-random generator

    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)           # pytorch (both CPU and CUDA)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False

        if set_torch_deterministic:
            configure_deterministic_mode()
    # fmt: on
    return seed


def seed_worker(worker_id: int) -> None:  # noqa: ARG001
    """
    Initializes random seeds for a worker based on its ID.

    This ensures that each worker, when used in parallel data loading,
    operates with a unique random state, which enhances reproducibility
    and reducing potential data overlap or bias in data loading processes.

    Parameters
    ----------
    worker_id : int
        The unique identifier for the data loader worker.

    References
    ----------
    - `PyTorch Worker <https://pytorch.org/docs/stable/notes/randomness.html#dataloader>`_

    Examples
    --------
    .. code-block:: python

        import torch
        from torch.utils.data import DataLoader

        train_dataset = ...

        g = torch.Generator()
        g.manual_seed(0)

        DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)  # noqa: NPY002
    random.seed(worker_seed)

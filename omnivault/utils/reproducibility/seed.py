"""https://github.com/mosaicml/composer/blob/dev/composer/utils/reproducibility.py"""

from __future__ import annotations

import os
import random
import warnings

import numpy as np
import torch
import torch.backends.cudnn

__all__ = ["seed_all"]


def configure_deterministic_mode() -> None:
    """
    See https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    and https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    """
    # fmt: off
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark        = False
    torch.backends.cudnn.deterministic    = True
    torch.backends.cudnn.enabled          = False

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # fmt: on
    warnings.warn(
        "Deterministic mode is activated. This will negatively impact performance.",
        category=UserWarning,
        stacklevel=2,
    )


def seed_all(
    seed: int = 1992,
    seed_torch: bool = True,
    set_torch_deterministic: bool = True,
) -> int:
    """
    Seed all random number generators.

    Parameters
    ----------
    seed : int, optional
        Seed number to be used, by default 1992.
    seed_torch : bool, optional
        Whether to seed PyTorch or not, by default True.

    Returns
    -------
    seed: int
        The seed number.
    """
    # fmt: off
    os.environ["PYTHONHASHSEED"] = str(seed)       # set PYTHONHASHSEED env var at fixed value
    np.random.default_rng(seed)                    # numpy pseudo-random generator
    random.seed(seed)                              # python's built-in pseudo-random generator

    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)           # pytorch (both CPU and CUDA)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

        if set_torch_deterministic:
            configure_deterministic_mode()
    # fmt: on
    return seed

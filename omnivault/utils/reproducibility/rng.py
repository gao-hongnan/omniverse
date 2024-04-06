"""This module provides customized random number generator utilities to enhance
reproducibility across different platforms.
"""
import random
from typing import Any, Dict

import numpy as np
import torch

__all__ = ["save_rng_state", "load_and_set_rng_state"]


def save_rng_state(save_dir: str, epoch_index: int) -> None:
    """Save the current state of various random number generators (RNGs) to a file,
    facilitating reproducible experimentation by allowing the exact state of RNGs to
    be restored later. This function is particularly useful in deep learning
    workflows where resuming training from a specific epoch requires the RNG state
    to be consistent with that epoch's initial state (i.e. resuming dataloaders,
    batch norm, dropout states, etc.).

    Parameters
    ----------
    save_dir : str
        The directory where the RNG state file will be saved. The function will
        create a file named `rng_state_epoch_{epoch_index}.pt` in this directory.
    epoch_index : int
        The epoch index corresponding to the RNG state being saved. This index is
        used to name the saved file, indicating the epoch with which the RNG state
        is associated.
    """
    rng_state = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),  # noqa: NPY002
        "torch_random_state": torch.random.get_rng_state(),
        "epoch_index": epoch_index,
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():  # type: ignore[no-untyped-call]
        # This will not be compatible with model parallelism
        rng_state["cuda"] = torch.cuda.get_rng_state()

    torch.save(rng_state, f"{save_dir}/rng_state_epoch_{epoch_index}.pt")


def load_and_set_rng_state(rng_state_path: str) -> Dict[str, Any]:
    """Load and set the state of various random number generators (RNGs) from a file,
    allowing the continuation of experiments or training processes from a specific
    point with reproducibility. This function restores the RNG states for Python's
    built-in `random` module, NumPy's random module, and PyTorch's global RNG,
    including CUDA's RNG state if applicable and present in the saved state.

    Parameters
    ----------
    rng_state_path : str
        The path to the file containing the saved RNG states. This file is expected
        to have been created by the `save_rng_state` function.

    Returns
    -------
    rng_state: Dict[str, Any]
        A dictionary containing the loaded RNG states. This includes states for
        Python's built-in `random` module (`python_random_state`), NumPy's random
        module (`numpy_random_state`), PyTorch's global RNG (`torch_random_state`),
        and, if applicable, PyTorch's CUDA RNG state (`cuda_rng_state`).
    """
    rng_state: Dict[str, Any] = torch.load(rng_state_path)

    random.setstate(rng_state["python_random_state"])
    np.random.set_state(rng_state["numpy_random_state"])  # noqa: NPY002
    torch.random.set_rng_state(rng_state["torch_random_state"])
    if "cuda_rng_state" in rng_state and torch.cuda.is_available():
        torch.cuda.set_rng_state(rng_state["cuda_rng_state"])
    return rng_state

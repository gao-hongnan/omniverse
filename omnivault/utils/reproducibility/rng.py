import random
from typing import Any

import numpy as np
import torch


def save_rng_state(save_dir: str, epoch_index: int) -> None:
    """Save the state of the random number generators, useful for resuming dataloaders,
    batch norm, dropout states."""
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


def load_and_set_rng_state(rng_state_path: str) -> Any:
    rng_state = torch.load(rng_state_path)

    random.setstate(rng_state["python_random_state"])
    np.random.set_state(rng_state["numpy_random_state"])  # noqa: NPY002
    torch.random.set_rng_state(rng_state["torch_random_state"])
    if "cuda_rng_state" in rng_state and torch.cuda.is_available():
        torch.cuda.set_rng_state(rng_state["cuda_rng_state"])
    return rng_state

from typing import Any, Tuple

import torch
from torch import nn

__all__ = ["total_trainable_parameters", "total_parameters", "compare_models"]


def total_trainable_parameters(module: nn.Module) -> int:
    """Returns the number of trainable parameters in the model."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def total_parameters(module: nn.Module) -> int:
    """Returns the total number of parameters in the model, including non-trainable."""
    return sum(p.numel() for p in module.parameters())


def compare_models(model_a: nn.Module, model_b: nn.Module) -> bool:
    """
    Compare two PyTorch models to check if they have identical parameters.

    Parameters
    ----------
    model_a : nn.Module
        The first model to compare.
    model_b : nn.Module
        The second model to compare.

    Returns
    -------
    bool
        Returns True if both models have identical parameters, False otherwise.
    """
    return all(
        torch.equal(param_a[1], param_b[1])
        for param_a, param_b in zip(model_a.state_dict().items(), model_b.state_dict().items())
    )


def compare_models_and_report_differences(model_a: nn.Module, model_b: nn.Module) -> Tuple[bool, Any]:
    """
    Compare two PyTorch models to check if they have identical parameters.

    Parameters
    ----------
    model_a : nn.Module
        The first model to compare.
    model_b : nn.Module
        The second model to compare.

    Returns
    -------
    Tuple[bool, Any]
        Returns a tuple with the first element as a boolean indicating if the models are identical.
        The second element is a dictionary containing the differences between the models if they are not identical.
    """
    model_a_dict = model_a.state_dict()
    model_b_dict = model_b.state_dict()

    if set(model_a_dict.keys()) != set(model_b_dict.keys()):
        # Early exit if model architectures are different (different sets of parameter keys)
        return False, {"error": "Models have different architectures and cannot be compared."}

    differences = {}
    for name in model_a_dict.keys():  # noqa: SIM118
        param_a = model_a_dict[name]
        param_b = model_b_dict[name]
        if not torch.equal(param_a, param_b):
            differences[name] = {
                "model_a": param_a.detach().cpu().numpy(),
                "model_b": param_b.detach().cpu().numpy(),
            }

    if differences:
        return False, differences
    else:
        return True, None

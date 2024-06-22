from typing import Any, Dict, List, Tuple

import torch
from torch import nn

__all__ = [
    "total_trainable_parameters",
    "total_parameters",
    "compare_models",
    "compare_models_and_report_differences",
    "get_named_modules",
]


def total_trainable_parameters(module: nn.Module) -> int:
    """Returns the number of trainable parameters in the model."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def total_parameters(module: nn.Module) -> int:
    """Returns the total number of parameters in the model, including non-trainable."""
    return sum(p.numel() for p in module.parameters())


def get_named_modules(model: nn.Module, **kwargs: Any) -> List[Dict[str, str]]:
    """Obtain a list of named modules in the model.

    Parameters
    ----------
    model : nn.Module
        The model to extract named modules from.
    **kwargs : Any
        Additional keyword arguments to pass to the `named_modules` method.

    Returns
    -------
    List[Dict[str, str]]
        A list of dictionaries containing the name and type of each module in the model.

    Examples
    --------
    >>> from torchvision.models import resnet18, ResNet18_Weights
    >>> backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    >>> named_modules = get_named_modules(backbone)
    """
    named_modules = []
    for module in model.named_modules(**kwargs):
        module_name, module_type = module
        named_modules.append({str(module_name): str(module_type)})
    return named_modules


def gather_weight_stats(model: nn.Module, **kwargs: Any) -> Dict[str, Dict[str, float]]:
    """Return the mean and standard deviation of weights and biases in the model. Sanity
    check to ensure that the weights and biases are initialized correctly.

    Parameters
    ----------
    model : nn.Module
        The model to extract weight statistics from.
    **kwargs : Any
        Additional keyword arguments to pass to the `named_modules` method.

    Returns
    -------
    Dict[str, Dict[str, float]]
        A dictionary containing the mean and standard deviation of weights and biases in the model.

    Examples
    --------
    >>> from torchvision.models import resnet18, ResNet18_Weights
    >>> backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    >>> stats = gather_weight_stats(backbone)
    """
    stats = {}
    for module in model.named_modules(**kwargs):
        module_name, module_type = module
        assert isinstance(module_type, nn.Module)
        if (
            hasattr(module_type, "weight")
            and isinstance(module_type.weight, torch.nn.Parameter)
            and module_type.weight is not None
        ):
            weight = module_type.weight.data
            weight_key = f"{module_name}+{str(module_type)}_weight"
            stats[weight_key] = {"w_mean": weight.mean().item(), "w_std": weight.std().item()}

        if (
            hasattr(module_type, "bias")
            and isinstance(module_type.bias, torch.nn.Parameter)
            and module_type.bias is not None
        ):
            bias = module_type.bias.data
            bias_key = f"{module_name}+{str(module_type)}_bias"
            stats[bias_key] = {"b_mean": bias.mean().item(), "b_std": bias.std().item()}
    return stats


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

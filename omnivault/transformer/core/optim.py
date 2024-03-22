from __future__ import annotations

from typing import Dict, List, Literal, Set, Tuple, Type

import torch
from torch import nn

from omnivault.transformer.modules.layers.normalization import LayerNorm


# FIXME: I have my own implementation of AddNorm, should I blacklist it or not?
def apply_weight_decay_to_different_param_groups(
    model: nn.Module, weight_decay: float
) -> List[Dict[Literal["params", "weight_decay"], List[torch.nn.Parameter] | float]]:
    """
    Categorizes parameters of a PyTorch model into two groups based on weight decay.

    Parameters that typically undergo weight decay (like Linear layer weights)
    and those that don't (like biases and LayerNorm weights) are separated.
    This categorization is useful for optimizers that apply weight decay differently
    to different types of parameters.

    Reference
    ----------
    PyTorch mingpt: https://github.com/pytorch/examples/blob/main/distributed/minGPT-ddp/mingpt/model.py

    Parameters
    ----------
    model : nn.Module
        The PyTorch model whose parameters are to be categorized.
    weight_decay : float
        The weight decay to be applied to the parameters that undergo weight decay.

    Returns
    -------
    optim_groups: List[Dict[str, torch.nn.Parameter]]
        A list containing two dictionaries, one for parameters to decay and
        one for parameters not to decay.

    Raises
    ------
    AssertionError
        If any parameter is found in both decay and no_decay sets.
    """

    decay: Set[str] = set()
    no_decay: Set[str] = set()
    whitelist_weight_modules: Tuple[Type[nn.Module], ...] = (nn.Linear,)
    blacklist_weight_modules: Tuple[Type[nn.Module], ...] = (nn.LayerNorm, nn.Embedding, LayerNorm)

    for module_name, module in model.named_modules():
        for parameter_name, _parameter in module.named_parameters():
            full_parameter_name = f"{module_name}.{parameter_name}" if module_name else parameter_name
            if parameter_name.endswith("bias"):
                # biases of all modules are not decayed
                no_decay.add(full_parameter_name)
            elif parameter_name.endswith("weight") and isinstance(module, whitelist_weight_modules):
                # weights of whitelisted modules are decayed
                decay.add(full_parameter_name)
            elif parameter_name.endswith("in_proj_weight"):
                # MHA projection layer, does not exist in my implementation
                decay.add(full_parameter_name)
            elif parameter_name.endswith("weight") and isinstance(module, blacklist_weight_modules):
                # weights of blacklisted modules are not decayed
                no_decay.add(full_parameter_name)
            elif (parameter_name.endswith("gamma") or parameter_name.endswith("beta")) and isinstance(
                module, LayerNorm
            ):
                # weights of LayerNorm modules are not decayed
                # TODO: why do I need to do this is because my custom LayerNorm has gamma and beta
                # as their "weight" and "bias" attributes, respectively.
                no_decay.add(full_parameter_name)
            elif parameter_name.endswith("pos_embed"):
                no_decay.add(full_parameter_name)

    param_dict = {parameter_name: parameter for parameter_name, parameter in model.named_parameters()}  # noqa: C416
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert not inter_params, f"Parameters {inter_params} are in both decay and no_decay sets."
    assert not (
        param_dict.keys() - union_params
    ), f"Parameters {param_dict.keys() - union_params} were not categorized."

    optim_groups: List[Dict[Literal["params", "weight_decay"], List[torch.nn.Parameter] | float]] = [
        {"params": [param_dict[parameter_name] for parameter_name in sorted(decay)], "weight_decay": weight_decay},
        {"params": [param_dict[parameter_name] for parameter_name in sorted(no_decay)], "weight_decay": 0.0},
    ]

    return optim_groups

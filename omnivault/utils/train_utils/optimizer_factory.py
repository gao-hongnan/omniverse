from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Type

from torch import nn

ALL_LAYERNORM_LAYERS = [nn.LayerNorm]
BLACKLISTED: List[str] = ["bias", "LayerNorm.weight", "LayerNorm.bias"]  # CHANGE AS YOU WISH


def get_parameter_names(model: nn.Module, forbidden_layer_types: Sequence[Type[nn.Module]]) -> List[str]:
    """
    Returns the names of the model parameters that are not inside a forbidden layer.

    Source: Huggingface
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def get_decay_parameter_names(model: nn.Module, blacklisted: Sequence[Type[nn.Module]] | None = None) -> List[str]:
    """
    Get all parameter names that weight decay will be applied to

    Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
    apply to those modules since this function only filter out instance of nn.LayerNorm.

    Source: Huggingface
    """
    blacklisted = blacklisted or ALL_LAYERNORM_LAYERS
    decay_parameters = get_parameter_names(model, blacklisted)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return decay_parameters


def get_huggingface_default_decay_and_optimizer_group(
    model: nn.Module, weight_decay: float = 0.01
) -> Tuple[List[str], List[Dict[str, str | float | List[nn.Parameter]]]]:
    """This function fetches the default decay parameters and optimizer group for Hugging Face models.
    Weight decay is 0.01 for most models.
    """
    HF_DEFAULT_DECAY = get_decay_parameter_names(model=model)
    HF_DEFAULT_OPTIMIZER_GROUP = [
        {
            "params": [
                parameter
                for parameter_name, parameter in model.named_parameters()
                if (parameter_name in HF_DEFAULT_DECAY and parameter.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                parameter
                for parameter_name, parameter in model.named_parameters()
                if parameter_name not in HF_DEFAULT_DECAY and parameter.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    return HF_DEFAULT_DECAY, HF_DEFAULT_OPTIMIZER_GROUP  # type: ignore[return-value]

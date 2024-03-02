from torch import nn


def total_trainable_parameters(module: nn.Module) -> int:
    """Returns the number of trainable parameters in the model."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def total_parameters(module: nn.Module) -> int:
    """Returns the total number of parameters in the model, including non-trainable."""
    return sum(p.numel() for p in module.parameters())

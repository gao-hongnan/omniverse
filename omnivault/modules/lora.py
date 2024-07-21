"""LoRA: Low-Rank Adaptation of Large Language Models.

References
----------
[1] https://pytorch.org/torchtune/stable/tutorials/lora_finetune.html
"""


from __future__ import annotations

import math
from typing import List

import torch
from pydantic import BaseModel, Field
from torch import nn


class LoraConfig(BaseModel):
    r: int = Field(..., description="Lora attention dimension (the 'rank').")
    lora_alpha: int = Field(..., description="The alpha parameter for Lora scaling.")
    lora_dropout: float = Field(..., description="The dropout probability for Lora layers.")
    target_modules: List[str] = Field(
        default=None,
        description=(
            "The names of the modules to apply the adapter to. If specified, only the modules with the specified "
            "names will be replaced. When passing a string, a regex match will be performed. When passing a list of "
            "strings, either an exact match will be performed or it is checked if the name of the module ends with any "
            "of the passed strings. If specified as 'all-linear', all linear/Conv1D modules are chosen, excluding the "
            "output layer. If not specified, modules are chosen according to the model architecture. If the architecture "
            "is unknown, an error will be raised—manual specification of target modules is required in such cases."
        ),
    )
    linear_bias: bool = Field(default=True, description="To include linear bias or not.")
    modules_to_save: List[str] = Field(
        default=None,
        description=(
            """List of modules apart from adapter layers to be set as
               trainable and saved in the final checkpoint."""
        ),
    )


def _lora_a_init_params(x: nn.Linear) -> None:
    """
    Initialize LoRA A weight to Kaiming uniform.
    """
    nn.init.kaiming_uniform_(x.weight, a=math.sqrt(5))


def _lora_b_init_params(x: nn.Linear) -> None:
    """
    Initialize LoRA B weight to zeros.
    """
    nn.init.zeros_(x.weight)


class LoRALinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()

        # These are the weights from the original pretrained model
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)  # weight shape=[out_dim, in_dim]

        # These are the new LoRA params. In general rank << in_dim, out_dim - do not put bias here
        self.lora_a = nn.Linear(in_features=in_dim, out_features=rank, bias=False)  # weight shape=[rank, in_dim]
        self.lora_b = nn.Linear(in_features=rank, out_features=out_dim, bias=False)  # weight shape=[out_dim, rank]

        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """See https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L119."""

        _lora_a_init_params(self.lora_a)
        _lora_b_init_params(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This would be the output of the original model
        frozen_out = x @ self.linear.weight.T
        if self.linear.bias is not None:
            frozen_out += self.linear.bias

        # lora_a projects inputs down to the much smaller self.rank,
        # then lora_b projects back up to the output dimension
        x = self.dropout(x)
        lora_out = (x @ self.lora_a.weight.T) @ self.lora_b.weight.T  # x @ lora_a @ lora_b
        # Finally, scale by the alpha parameter (normalized by rank)
        # and add to the original model's outputs
        return frozen_out + (self.alpha / self.rank) * lora_out


def apply_lora_to_base_model(
    model: nn.Module, rank: int, alpha: float, dropout: float, target_modules: List[str] | None = None
) -> None:
    """Recursively apply LoRA to a model. Only supports applying on `nn.Linear` layers."""

    for module_name, module in model.named_children():
        if isinstance(module, nn.Linear):
            if target_modules is None or any(target in module_name for target in target_modules):
                setattr(
                    model,
                    module_name,
                    LoRALinear(
                        in_dim=module.in_features,
                        out_dim=module.out_features,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                        bias=module.bias is not None,
                    ),
                )
        else:
            # Recursively apply LoRA to children modules
            apply_lora_to_base_model(
                model=module, rank=rank, alpha=alpha, dropout=dropout, target_modules=target_modules
            )

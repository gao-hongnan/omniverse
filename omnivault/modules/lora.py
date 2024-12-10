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
    """LoRA Linear layer."""

    def __init__(self, original_linear: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        """Initialize the `LoRALinear` layer.

        Parameters
        ----------
        original_linear : nn.Linear
            The original linear layer from the pretrained
        rank : int
            The rank of the LoRA layer.
        alpha : float
            The alpha parameter for LoRA scaling.
        dropout : float
            The dropout probability for the LoRA layer.
        """
        super().__init__()

        # These are the weights from the original pretrained model
        self.linear = original_linear  # weight shape=[out_dim, in_dim]

        in_dim = self.linear.in_features
        out_dim = self.linear.out_features

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
        """Forward pass of the `LoRALinear` layer."""
        frozen_out = x @ self.linear.weight.T  # This would be the output of the original model
        if self.linear.bias is not None:
            frozen_out += self.linear.bias

        # lora_a projects inputs down to the much smaller self.rank,
        # then lora_b projects back up to the output dimension
        x = self.dropout(x)
        lora_out = x @ (self.lora_a.weight.T @ self.lora_b.weight.T)  # [B, T, D1] @ [D1, R] @ [R, D2] = [D1, D2]
        # Finally, scale by the alpha parameter (normalized by rank)
        # and add to the original model's outputs
        return frozen_out + (self.alpha / self.rank) * lora_out

    @torch.no_grad()
    def _merge(self) -> nn.Linear:
        """
        Merge the LoRA layers to the original linear layer.
        """

        # (gate_proj): Linear(in_features=1024, out_features=2816, bias=False) -> weight = [2816, 1024]
        # [1024, R] @ [R, 2816]
        # torch.Size([1024, 2816]) torch.Size([2816, 1024])

        lora_weight = self.lora_a.weight.T @ self.lora_b.weight.T  # [D1, R] @ [R, D2] = [D1, D2]
        lora_weight = lora_weight.to(self.linear.weight.device)
        self.linear.weight += (self.alpha / self.rank) * lora_weight.T
        return self.linear


def merge_and_unload_model(model: nn.Module) -> nn.Module:
    """Recursively merge LoRA layers back into the original Linear layers in
    the model and unload LoRA parameters."""
    for module_name, module in model.named_children():
        if isinstance(module, LoRALinear):
            merged_linear = module._merge()
            setattr(model, module_name, merged_linear)
        else:
            merge_and_unload_model(module)
    return model


def apply_lora_to_base_model(
    model: nn.Module, rank: int, alpha: float, dropout: float, target_modules: List[str] | None = None
) -> None:
    """Recursively apply LoRA to a model. Only supports applying on `nn.Linear` layers.

    In the `if` condition, we first check if the module is an instance of
    `nn.Linear`. If it is, we then check if the `target_modules` is specified
    by user, if it is not, then `if target_modules is None` will return `True`
    and we apply LoRA to the module because we assume that the user wants to
    apply LoRA to all `nn.Linear` layers. If the `target_modules` is specified,
    then `if target_modules is None` will return `False` and we will check the
    second condition `any(target in module_name for target in target_modules)`
    which will return `True` if any of the target modules are in the module name.
    """
    for module_name, module in model.named_children():
        if isinstance(module, nn.Linear):
            if target_modules is None or any(target in module_name for target in target_modules):
                setattr(
                    model,
                    module_name,
                    LoRALinear(
                        original_linear=module,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                    ),
                )
        else:
            # Recursively apply LoRA to children modules
            apply_lora_to_base_model(
                model=module, rank=rank, alpha=alpha, dropout=dropout, target_modules=target_modules
            )

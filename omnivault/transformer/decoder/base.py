"""Base classes for decoders in transformer-like architectures. Template design pattern."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from omnivault._types._alias import NotGiven
from omnivault._types._sentinel import NOT_GIVEN
from omnivault.transformer.config.decoder import DecoderConfig


class BaseDecoderBlock(ABC, nn.Module):
    """
    Abstract base class for a decoder block in a transformer-like architecture.
    """

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self,
        z: torch.Tensor,  # that's tgt in torch code base
        *,  # force keyword only arguments to prevent errors
        encoder_hidden_states: torch.Tensor | NotGiven = NOT_GIVEN,  # that's memory in torch code base
        encoder_hidden_states_masks: torch.BoolTensor | NotGiven = NOT_GIVEN,  # that's memory_mask in torch code base
        target_masks: torch.BoolTensor | NotGiven = NOT_GIVEN,  # that's tgt_mask in torch code base
    ) -> torch.Tensor:
        """
        Performs one decoder *block* forward pass given final encoder hidden states, the previous block's output, and
        attention masks.

        N = batch size
        S = source sequence length
        T = target sequence length
        E = embedding dimensionality
        V = vocabulary size

        :param x: Previous decoder block's output. Shape: (N, T, E)
        :param encoder_hidden_states: The encoder's final (contextualized) token embeddings. Shape: (N, S, E)
        :param src_padding_mask: An attention mask to ignore pad-tokens in the source input. Shape (N, S)
        :param future_mask: An attention mask to ignore future-tokens in the target input. Shape (T, T)
        :return: Updated, contextualized token embeddings. Shape (N, T, E)
        """


class BaseDecoder(nn.Module, ABC):
    """
    Abstract base class for a decoder in a transformer-like architecture.
    """

    def __init__(
        self,
        config: DecoderConfig,
    ) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self,
        input_tokens: torch.LongTensor,
        *,  # force keyword only arguments to prevent errors
        target_padding_masks: torch.BoolTensor | NotGiven = NOT_GIVEN,
        future_masks: torch.BoolTensor | NotGiven = NOT_GIVEN,
        encoder_hidden_states: torch.Tensor | NotGiven = NOT_GIVEN,  # that's memory in torch code base
        encoder_hidden_states_masks: torch.BoolTensor | NotGiven = NOT_GIVEN,  # that's memory_mask in torch code base
    ) -> torch.FloatTensor: ...

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes weights of the given module using Xavier uniform initialization."""
        for p in module.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

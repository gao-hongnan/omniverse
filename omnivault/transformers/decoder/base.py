"""Base classes for decoders in transformer-like architectures.
Template Design Pattern.

The `memory_mask` in the context of a Transformer decoder is not the same as a future mask, also known as a causal mask or look-ahead mask. These masks serve different purposes:

1. **Memory Mask (`memory_mask`):**
   - The `memory_mask` is used in the decoder to mask the encoder's output (`memory`).
   - Its purpose is to control which parts of the encoder output the decoder can attend to. For instance, it can be used to prevent the decoder from attending to padding tokens or other specific positions in the encoder output.
   - It's not directly related to the concept of preventing future information leakage.

2. **Future Mask / Causal Mask:**
   - A future mask or causal mask is used in autoregressive models like GPT to prevent a position from attending to subsequent positions. This is crucial in generation tasks where the model should not have access to future tokens in the sequence it is generating.
   - In the decoder of a Transformer model (like in the original Transformer or BERT), this mask ensures that during the self-attention phase, a token can only attend to itself and preceding tokens, not to any future tokens.

### Key Differences:
- **Context of Use:** The `memory_mask` is applied to the output of the encoder when it is being processed by the decoder. The future mask is used within the self-attention mechanism of the decoder to ensure causality in the sequence generation.
- **Purpose:** The `memory_mask` controls attention to the encoder's output, while the future mask enforces causality by preventing forward-looking attention within the sequence being processed by the decoder.

### Conclusion:
The `memory_mask` and the future mask are different components serving distinct purposes in a Transformer model. The former relates to how the decoder interacts with the encoder's output, and the latter is about maintaining the autoregressive property in sequence generation.
"""

from typing import Optional

import torch
from torch import nn

from omnivault.transformers.config.decoder import DecoderConfig


class BaseDecoderBlock(nn.Module):
    """
    Abstract base class for a decoder block in a transformer-like architecture.
    """

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        z: torch.Tensor,  # that's tgt in torch code base
        encoder_hidden_states: Optional[
            torch.Tensor
        ] = None,  # that's memory in torch code base
        encoder_hidden_states_masks: Optional[
            torch.BoolTensor
        ] = None,  # that's memory_mask in torch code base
        target_masks: Optional[
            torch.BoolTensor
        ] = None,  # that's tgt_mask in torch code base
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


class BaseDecoder(nn.Module):
    """
    Abstract base class for a decoder in a transformer-like architecture.
    """

    def __init__(
        self,
        config: DecoderConfig,
    ) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        input_tokens: torch.LongTensor,
        target_padding_masks: Optional[torch.BoolTensor] = None,
        future_masks: Optional[torch.BoolTensor] = None,
        encoder_hidden_states: Optional[
            torch.Tensor
        ] = None,  # that's memory in torch code base
        encoder_hidden_states_masks: Optional[
            torch.BoolTensor
        ] = None,  # that's memory_mask in torch code base
    ) -> torch.FloatTensor:
        ...

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

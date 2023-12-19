from __future__ import annotations

from typing import cast, overload

import torch
from torch import nn

from omnivault._types._alias import NotGiven
from omnivault._types._sentinel import NOT_GIVEN
from omnivault.transformer.config.decoder import DecoderConfig
from omnivault.transformer.decoder.base import BaseDecoder, BaseDecoderBlock
from omnivault.transformer.modules.attention.core import MultiHeadedAttention
from omnivault.transformer.modules.layers.addnorm import AddNorm
from omnivault.transformer.modules.layers.mlp import PositionwiseFeedForward


class GPTDecoderBlock(BaseDecoderBlock):
    """GPTDecoderBlock focuses on masked self-attention and feed-forward layers.

    The architecture follows the GPT-style decoder, which only has masked
    self-attention and position-wise feed-forward layers, omitting the
    encoder-decoder cross-attention.
    """

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__(config)
        # fmt: off
        self.masked_self_attention_mha = MultiHeadedAttention(**config.decoder_block.masked_self_attention_mha.__dict__)
        # self.encoder_decoder_cross_attention_mha = MultiHeadedAttention(**config.decoder.encoder_decoder_cross_attention_mha)

        self.feed_forward              = PositionwiseFeedForward(**config.decoder_block.feed_forward.__dict__)

        self.add_norm_1                = AddNorm(**config.decoder_block.add_norm_1.__dict__)
        self.add_norm_2                = AddNorm(**config.decoder_block.add_norm_2.__dict__)

        # self.feed_forward.register_forward_hook(forward_hook)
        # fmt: on

    def forward(
        self,
        z: torch.Tensor,  # that's tgt in torch code base
        *,
        encoder_hidden_states: torch.Tensor | NotGiven = NOT_GIVEN,  # that's memory in torch code base
        encoder_hidden_states_masks: torch.BoolTensor | NotGiven = NOT_GIVEN,  # that's memory_mask in torch code base
        target_masks: torch.BoolTensor | NotGiven = NOT_GIVEN,  # that's tgt_mask in torch code base
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z:              Input sequence.
                        type:  torch.Tensor
                        shape: (B, S or T, D)
        target_masks:   Target mask.
                        type:  torch.BoolTensor
                        shape: (B, 1, S or T, S or T)

        Returns
        -------
        z:              Output tensor after masked self-attention and feed-forward layers.
                        type:  torch.Tensor
                        shape: (B, S or T, D)
        """
        assert encoder_hidden_states is NOT_GIVEN, "GPTDecoderBlock does not have encoder-decoder cross-attention"
        assert encoder_hidden_states_masks is NOT_GIVEN, "GPTDecoderBlock does not have encoder-decoder cross-attention"
        assert target_masks is not NOT_GIVEN

        z = self.add_norm_1(
            z,
            lambda z: self.masked_self_attention_mha(query=z, key=z, value=z, mask=target_masks),
        )
        z = self.add_norm_2(z, self.feed_forward)
        return z


class GPTDecoder(BaseDecoder):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__(config)
        # fmt: off
        self.d_model       : int           = config.d_model
        self.tok_embed     : nn.Embedding  = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed     : nn.Parameter  = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_model))
        self.decoder_blocks: nn.ModuleList = nn.ModuleList([GPTDecoderBlock(config) for _ in range(config.num_decoder_blocks)]) # PyTorch did not make ModuleList a proper container, maybe open a PR to make it inherit Generic[T]???

        self.dropout       : nn.Dropout    = nn.Dropout(config.dropout)
        self.layer_norm    : nn.LayerNorm  = nn.LayerNorm(config.d_model)
        self.head          : nn.Linear     = nn.Linear(config.d_model, config.vocab_size)  # last layer
        # fmt: on

        self._reset_parameters()

    # @overload
    # def create_target_masks(self, target_padding_masks: torch.Tensor, future_masks: torch.Tensor) -> torch.BoolTensor:
    #     ...

    # @overload
    # def create_target_masks(
    #     self, target_padding_masks: torch.Tensor, future_masks: torch.Tensor | None
    # ) -> torch.BoolTensor:
    #     ...

    # @overload
    # def create_target_masks(
    #     self, target_padding_masks: torch.Tensor | None, future_masks: torch.Tensor
    # ) -> torch.BoolTensor:
    #     ...

    # @overload
    # def create_target_masks(
    #     self,
    #     target_padding_masks: torch.Tensor | None,
    #     future_masks: torch.Tensor | None,
    # ) -> torch.BoolTensor:
    #     ...

    # def create_target_masks(
    #     self,
    #     target_padding_masks: torch.Tensor | None,
    #     future_masks: torch.Tensor | None,
    # ) -> torch.BoolTensor:
    #     if target_padding_masks is None and future_masks is None:
    #         raise ValueError("At least one of target_padding_masks or future_masks must not be None")

    #     if target_padding_masks is None:
    #         assert future_masks is not None  # for mypy
    #         target_padding_masks = torch.ones_like(future_masks, dtype=torch.bool)

    #     if future_masks is None:
    #         assert target_padding_masks is not None  # for mypy
    #         future_masks = torch.ones_like(target_padding_masks, dtype=torch.bool)

    #     return torch.logical_and(target_padding_masks, future_masks).bool()  # type: ignore[return-value]

    @overload
    def create_target_masks(
        self, target_padding_masks: torch.BoolTensor, future_masks: torch.BoolTensor
    ) -> torch.BoolTensor:
        ...

    @overload
    def create_target_masks(self, target_padding_masks: torch.BoolTensor, future_masks: NotGiven) -> torch.BoolTensor:
        ...

    @overload
    def create_target_masks(self, target_padding_masks: NotGiven, future_masks: torch.BoolTensor) -> torch.BoolTensor:
        ...

    @overload
    def create_target_masks(self, target_padding_masks: NotGiven, future_masks: NotGiven) -> torch.BoolTensor:
        ...

    def create_target_masks(
        self,
        target_padding_masks: torch.BoolTensor | NotGiven = NOT_GIVEN,
        future_masks: torch.BoolTensor | NotGiven = NOT_GIVEN,
    ) -> torch.BoolTensor:
        """
        Creates a combined target mask for use in decoder layers, based on provided
        target padding masks and future masks. If either mask is not provided (NotGiven),
        a default mask is created using `torch.ones_like` to ensure shape compatibility
        and neutral behavior in subsequent operations.

        The default mask created by `torch.ones_like` acts as a placeholder that
        allows operations to proceed without altering the behavior that the mask
        would impose. This is particularly useful when the absence of a mask should
        not lead to masking out or altering any data, but rather to a 'pass-through'
        behavior. For example, in the case of target padding masks, the default mask
        allows the model to attend to all tokens in the target sequence, since the
        default mask is all ones and thus does not mask out any tokens. What this
        means is if the user does not provide a target padding mask, the model will
        not mask out any tokens in the target sequence, which is the same behavior
        as if the user had provided a target padding mask of all ones.

        Parameters
        ----------
        target_padding_masks : torch.BoolTensor | NotGiven, optional
            The mask for the target padding, indicating which elements of the target
            should be masked out. If not provided, a default mask of ones is created
            that does not mask out anything.

        future_masks : torch.BoolTensor | NotGiven, optional
            The mask for future tokens, typically used in self-attention mechanisms
            to prevent the model from 'seeing' future tokens. If not provided, a
            default mask of ones is created that does not prevent attending to future
            tokens.

        Returns
        -------
        torch.BoolTensor
            A combined boolean mask that is the logical AND of the target padding mask
            and future mask. The shape of the returned mask matches the input masks.

        Raises
        ------
        ValueError
            If both target_padding_masks and future_masks are NotGiven, a ValueError
            is raised since at least one mask is required for the operation.
        """
        if target_padding_masks is NOT_GIVEN and future_masks is NOT_GIVEN:
            raise ValueError("At least one of target_padding_masks or future_masks must not be None")

        # FIXME: CAN SOMEONE PLEASE HELP ME WITH TYPING HERE?? I AM SO STUCK IN CASTING HELL.
        if target_padding_masks is NOT_GIVEN:
            target_padding_masks = cast(
                torch.BoolTensor, torch.ones_like(cast(torch.Tensor, future_masks), dtype=torch.bool)
            )

        if future_masks is NOT_GIVEN:
            future_masks = cast(
                torch.BoolTensor, torch.ones_like(cast(torch.Tensor, target_padding_masks), dtype=torch.bool)
            )

        return cast(
            torch.BoolTensor,
            torch.logical_and(cast(torch.Tensor, target_padding_masks), cast(torch.Tensor, future_masks)).bool(),
        )

    def forward(
        self,
        input_tokens: torch.LongTensor,
        *,  # force keyword only arguments to prevent errors
        target_padding_masks: torch.BoolTensor | NotGiven = NOT_GIVEN,
        future_masks: torch.BoolTensor | NotGiven = NOT_GIVEN,
        encoder_hidden_states: torch.Tensor | NotGiven = NOT_GIVEN,  # that's memory in torch code base
        encoder_hidden_states_masks: torch.BoolTensor | NotGiven = NOT_GIVEN,  # that's memory_mask in torch code base
    ) -> torch.FloatTensor:
        """
        Notations
        ---------
        B:      Batch size
        S or L: Source sequence length
        T or L: Target sequence length
        D:      Embedding dimension
        V:      Vocabulary size (Class size)

        Parameters
        ----------
        input_tokens:           Input sequence.
                                type:  torch.Tensor
                                shape: (B, S or T)
        target_padding_masks:   Target padding mask.
                                type:  torch.BoolTensor
                                shape: (B, 1, S or T, S or T)
        future_masks:           Future mask.
                                type:  torch.BoolTensor
                                shape: (B, 1, S or T, S or T)

        Variables
        ---------
        z:                      Input sequence after token and position embedding.
                                type:  torch.Tensor
                                shape: (B, S or T, D)
        target_masks:           Target mask.
                                type:  torch.BoolTensor
                                shape: (B, 1, S or T, S or T)
        logits:                 Output logits.
                                type:  torch.FloatTensor
                                shape: (B, S or T, V)
        """
        assert encoder_hidden_states is NOT_GIVEN, "GPTDecoderBlock does not have encoder-decoder cross-attention"
        assert encoder_hidden_states_masks is NOT_GIVEN, "GPTDecoderBlock does not have encoder-decoder cross-attention"

        # fmt: off
        seq_len     : int              = input_tokens.size(1)
        target_masks: torch.BoolTensor = self.create_target_masks(target_padding_masks, future_masks)

        z = self.tok_embed(input_tokens) # * math.sqrt(self.d_model) for better optimization landscape
        z = z + self.pos_embed[:, :seq_len, :]
        z = self.dropout(z)

        for decoder_block in self.decoder_blocks:
            z  = decoder_block(z, target_masks=target_masks)

        z      = self.layer_norm(z)
        logits: torch.FloatTensor = self.head(z)
        # fmt: on
        return logits

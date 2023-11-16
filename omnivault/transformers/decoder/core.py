from typing import List, Optional

import torch
from torch import nn

from omnivault.transformers.decoder.base import BaseDecoder, BaseDecoderBlock
from omnivault.transformers.modules.attention.core import MultiHeadedAttention
from omnivault.transformers.modules.layers.addnorm import AddNorm
from omnivault.transformers.modules.layers.mlp import PositionwiseFeedForward


class GPTDecoderBlock(BaseDecoderBlock):
    """GPTDecoderBlock focuses on masked self-attention and feed-forward layers.

    The architecture follows the GPT-style decoder, which only has masked
    self-attention and position-wise feed-forward layers, omitting the
    encoder-decoder cross-attention.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        # fmt: off
        self.masked_self_attention_mha = MultiHeadedAttention(**config.decoder.masked_self_attention_mha.__dict__)
        # self.encoder_decoder_cross_attention_mha = MultiHeadedAttention(**config.decoder.encoder_decoder_cross_attention_mha)

        self.feed_forward              = PositionwiseFeedForward(**config.decoder.feed_forward.__dict__)

        self.add_norm_1                = AddNorm(**config.decoder.add_norm_1.__dict__)
        self.add_norm_2                = AddNorm(**config.decoder.add_norm_2.__dict__)

        # self.feed_forward.register_forward_hook(forward_hook)
        # fmt: on

    def forward(
        self, z: torch.Tensor, target_masks: Optional[torch.BoolTensor] = None
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
        z = self.add_norm_1(
            z,
            lambda z: self.masked_self_attention_mha(
                query=z, key=z, value=z, mask=target_masks
            ),
        )
        z = self.add_norm_2(z, self.feed_forward)
        return z


class GPTDecoder(BaseDecoder):
    def __init__(self, config):
        super().__init__(config)
        # fmt: off
        self.d_model       : int                   = config.d_model
        self.tok_embed     : nn.Embedding          = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed     : nn.Parameter          = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_model))
        self.decoder_blocks: List[GPTDecoderBlock] = nn.ModuleList([GPTDecoderBlock(config) for _ in range(config.num_layers)])

        self.dropout       : nn.Dropout            = nn.Dropout(config.dropout)
        self.layer_norm    : nn.LayerNorm          = nn.LayerNorm(config.d_model)
        self.head          : nn.Linear             = nn.Linear(config.d_model, config.vocab_size)  # last layer
        # fmt: on

        self._reset_parameters()

    def forward(
        self,
        input_tokens: torch.LongTensor,
        target_padding_masks: Optional[torch.BoolTensor] = None,
        future_masks: Optional[torch.BoolTensor] = None,
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
        # fmt: off
        seq_len     : int              = input_tokens.size(1)
        target_masks: torch.BoolTensor = torch.logical_and(target_padding_masks, future_masks)

        z = self.tok_embed(input_tokens) # * math.sqrt(self.d_model) for better optimization landscape
        z = z + self.pos_embed[:, :seq_len, :]
        z = self.dropout(z)

        for decoder_block in self.decoder_blocks:
            z  = decoder_block(z, target_masks)

        z      = self.layer_norm(z)
        logits = self.head(z)
        # fmt: on
        return logits

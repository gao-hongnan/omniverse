"""We use dataclass here for easy instantiating with hydra"""

from pydantic import BaseModel, Field
from torch import nn

from omnivault.transformer.modules.attention.base import Attention


class MultiHeadedAttentionConfig(BaseModel):
    attention: Attention
    d_model: int
    H: int
    dropout: float = 0.1

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


# TODO: add `field_validator` such that if `d_ff` is `None`, then `d_ff` is set to `4 * d_model`.
class PositionwiseFeedForwardConfig(BaseModel):
    d_model: int
    d_ff: int
    activation: nn.Module = Field(
        default=nn.GELU(approximate="tanh")
    )  # NOTE: https://github.com/facebookresearch/xformers/issues/759
    dropout: float = 0.1
    bias: bool = True

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


class AddNormConfig(BaseModel):
    feature_dim: int
    dropout: float


class DecoderBlockConfig(BaseModel):
    masked_self_attention_mha: MultiHeadedAttentionConfig
    feed_forward: PositionwiseFeedForwardConfig
    add_norm_1: AddNormConfig
    add_norm_2: AddNormConfig


class DecoderConfig(BaseModel):
    d_model: int
    vocab_size: int
    context_length: int  # NOTE: alias=max_seq_len,block_size
    num_decoder_blocks: int
    dropout: float
    decoder_block: DecoderBlockConfig

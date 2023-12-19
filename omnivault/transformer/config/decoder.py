"""We use dataclass here for easy instantiating with hydra"""
from dataclasses import dataclass, field

from torch import nn

from omnivault.transformer.modules.attention.base import Attention


@dataclass
class MultiHeadedAttentionConfig:
    attention: Attention
    d_model: int
    H: int
    dropout: float = 0.1


@dataclass
class PositionwiseFeedForwardConfig:
    d_model: int
    d_ff: int
    activation: nn.Module = field(default_factory=nn.ReLU())
    dropout: float = 0.1
    bias: bool = True


@dataclass
class AddNormConfig:
    feature_dim: int
    dropout: float


@dataclass
class DecoderBlockConfig:
    masked_self_attention_mha: MultiHeadedAttentionConfig
    feed_forward: PositionwiseFeedForwardConfig
    add_norm_1: AddNormConfig
    add_norm_2: AddNormConfig


@dataclass
class DecoderConfig:
    d_model: int
    vocab_size: int
    max_seq_len: int
    num_decoder_blocks: int
    dropout: float
    decoder_block: DecoderBlockConfig

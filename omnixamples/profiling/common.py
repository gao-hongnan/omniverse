from __future__ import annotations

from typing import Literal, Tuple, cast

import torch
from pydantic import BaseModel
from torch import nn

from omnivault.modules.activation import GELU, SoftmaxStable
from omnivault.transformer.modules.layers.normalization import RMSNorm

__tagged__ = "This code tags to `30d963e` of cs336-stanford-spring2024-assignment1-gpt-from-scratch."
__reference__ = ["https://github.com/marcelroed/spring2024-assignment2-systems/blob/master/writeup.pdf"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class General(BaseModel):
    batch_size: int = 16
    seed: int = 20230310


class ProfilerConfig(BaseModel):
    computation: Literal["forward", "backward", "forward_backward"]
    warmup_steps: int | None = None
    profile_steps: int
    mixed_precision: bool = False


class GPTConfig(BaseModel):
    approximate: Literal["tanh"] | None = None
    activation_name: Literal["gelu"] = "gelu"
    d_model: int
    d_ff: int | None = None
    num_heads: int
    context_length: int
    attn_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    bias: bool = False
    vocab_size: int
    num_blocks: int
    token_position_pdrop: float = 0.0
    weight_tie: bool = False


class PositionwiseFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        bias: bool = False,
        activation_name: Literal["gelu"] = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model
        self.bias = bias  # bias False in this exercise
        self.activation_name = activation_name
        self.dropout = dropout

        self.ffn = nn.ModuleDict(
            {
                # incoming `B x T x D` and we are interested in `T x D` so weight is `D x d_ff`
                # so that `Z @ W1 -> (T x D) @ (D x d_ff)`
                "context_fc": nn.Linear(in_features=self.d_model, out_features=self.d_ff, bias=self.bias),
                "activation": self.activation,
                # apply dropout after activation for random lights out
                "dropout": nn.Dropout(p=self.dropout, inplace=False),
                # incoming is Z @ W1 -> T x d_ff -> (T x d_ff) @ (d_ff x D) project back to D
                "context_projection": nn.Linear(in_features=self.d_ff, out_features=self.d_model, bias=self.bias),
            }
        )

    @property
    def activation(self) -> nn.Module:
        if self.activation_name == "gelu":
            activation = GELU(approximate=None)  # no approx using tanh
        else:
            raise ValueError(f"Unsupported activation: {self._activation}")
        return activation

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # fmt: off
        z = self.ffn["context_fc"](z)           # Z @ W1 = [B, T, D] @ [D, d_ff] = [B, T, d_ff]
        z = self.ffn["activation"](z)           # \sigma(Z @ W1) = [B, T, d_ff]
        z = self.ffn["dropout"](z)              # \dropout(\sigma(Z @ W1)) = [B, T, d_ff]
        z = self.ffn["context_projection"](z)   # \dropout(\sigma(Z @ W1)) @ W2 = [B, T, d_ff] @ [d_ff, D] = [B, T, D]
        # fmt: on
        return z


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.BoolTensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # fmt: off
        T, d_q = query.size(-2), query.size(-1)

        attention_scores  = torch.matmul(query, key.transpose(dim0=-2, dim1=-1)) / torch.sqrt(torch.tensor(d_q).float())        # Q @ K.T = [B, H, T, d_q] @ [B, H, d_q, T] = [B, H, T, T]

        if mask is not None:
            mask = mask[:, :, :T, :T] # type: ignore[assignment]
            attention_scores  = attention_scores.masked_fill(mask == 1, float("-inf")) if mask is not None else attention_scores    # [B, H, T, T]

        softmax           = SoftmaxStable(dim=-1)
        attention_weights = softmax(attention_scores)               # [B, H, T, T]
        attention_weights = self.dropout(attention_weights)         # [B, H, T, T]

        context_vector    = torch.matmul(attention_weights, value)  # [B, H, T, T] @ [B, H, T, d_v] = [B, H, T, d_v]
        # fmt: on
        return context_vector, attention_weights


class CausalMultiHeadSelfAttention(nn.Module):
    context_vector: torch.Tensor
    attention_weights: torch.Tensor

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        context_length: int,
        attn_pdrop: float = 0.0,  # pdrop means prob of dropout
        resid_pdrop: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.H = num_heads
        self.context_length = context_length
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.bias = bias

        self.W_Q = nn.Linear(in_features=self.d_model, out_features=self.d_model, bias=self.bias)
        self.W_K = nn.Linear(in_features=self.d_model, out_features=self.d_model, bias=self.bias)
        self.W_V = nn.Linear(in_features=self.d_model, out_features=self.d_model, bias=self.bias)

        # alias of W_O
        self.context_projection = nn.Linear(in_features=self.d_model, out_features=self.d_model, bias=self.bias)

        # regularization
        self.resid_dropout = nn.Dropout(self.resid_pdrop)

        self.attention = ScaledDotProductAttention(dropout=self.attn_pdrop)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        # register buffer cause not learnable weights
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones((self.context_length, self.context_length)).bool(),
                diagonal=1,
            ).view(1, 1, self.context_length, self.context_length),
        )

    def forward(self, *, z: torch.Tensor) -> torch.Tensor:
        B, T, D = z.size()

        # fmt: off
        Q: torch.Tensor = self.W_Q(z).contiguous() # Z @ W_Q = [B, T, D] @ [D, D] = [B, T, D]
        K: torch.Tensor = self.W_K(z).contiguous() # Z @ W_K = [B, T, D] @ [D, D] = [B, T, D]
        V: torch.Tensor = self.W_V(z).contiguous() # Z @ W_V = [B, T, D] @ [D, D] = [B, T, D]

        Q = Q.view(B, T, self.H, D // self.H).transpose(dim0=1, dim1=2) # [B, T, D] -> [B, T, H, D // H] -> [B, H, T, D//H]
        K = K.view(B, T, self.H, D // self.H).transpose(dim0=1, dim1=2)
        V = V.view(B, T, self.H, D // self.H).transpose(dim0=1, dim1=2)

        # Now pass them to self attention
        self.context_vector, self.attention_weights = self.attention(query=Q, key=K, value=V, mask=self.causal_mask) # ([B, H, T, D // H], [B, H, T, T])
        assert isinstance(self.context_vector, torch.Tensor) # do this for type hint in IDE

        # Now context vector is shape [B, H, T, D // H] but we want [B, T, D] to matmul with W_O/context_projection
        self.context_vector = self.context_vector.transpose(dim0=1, dim1=2).contiguous().view(B, T, D) # merge all heads together
        # fmt: on

        projected_context_vector: torch.Tensor = self.resid_dropout(
            self.context_projection(self.context_vector)  # [B, T, D] @ [D, D] = [B, T, D]
        )
        return projected_context_vector


class GPTBlock(nn.Module):
    def __init__(
        self,
        config: GPTConfig,
    ) -> None:
        super().__init__()

        self.rmns_1 = RMSNorm(d_model=config.d_model, eps=1e-5)
        self.attn = CausalMultiHeadSelfAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            context_length=config.context_length,
            attn_pdrop=config.attn_pdrop,
            resid_pdrop=config.resid_pdrop,
            bias=config.bias,
        )
        self.rmns_2 = RMSNorm(d_model=config.d_model, eps=1e-5)
        self.ffn = PositionwiseFeedForward(
            d_model=config.d_model,
            d_ff=config.d_ff,
            bias=config.bias,
            activation_name=config.activation_name,
            dropout=config.resid_pdrop,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z + self.attn(z=self.rmns_1(z))
        z = z + self.ffn(self.rmns_2(z))
        return z


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        self.config = config
        self.d_model = config.d_model
        self.num_blocks = config.num_blocks
        self.vocab_size = config.vocab_size

        self.blocks = nn.ModuleList([GPTBlock(config=config) for _ in range(self.num_blocks)])

        self.backbone = nn.ModuleDict(
            dict(  # noqa: C408
                token_embeddings=nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model),
                position_embeddings=nn.Embedding(num_embeddings=config.context_length, embedding_dim=self.d_model),
                dropout=nn.Dropout(p=config.token_position_pdrop),
                layers=self.blocks,
                ln_final=RMSNorm(d_model=self.d_model, eps=1e-5),
            )
        )
        self.head = nn.Linear(in_features=self.d_model, out_features=self.vocab_size, bias=config.bias)

        self.apply(self._init_weights)

        context_projections = "context_projection.weight"
        # apply special scaled init to the residual projections, per GPT-2 paper
        for parameter_name, parameter in self.named_parameters():
            # NOTE: W_O is also projection but I did not have foresight to name it as such.
            if parameter_name.endswith(context_projections):
                mean = 0.0
                std_dev = 0.02 / torch.sqrt(torch.tensor(2 * config.num_blocks, dtype=torch.float))
                torch.nn.init.normal_(parameter, mean=mean, std=std_dev)

        if config.weight_tie:
            self.backbone.token_embeddings.weight = self.head.weight

    def crop_context_length(self, context_length: int) -> None:
        # NOTE: conveniently took Karpathy's implementation here for cropping
        assert context_length <= self.config.context_length
        self.config.context_length = context_length  # update config

        self.backbone.position_embeddings.weight = nn.Parameter(
            self.backbone.position_embeddings.weight[:context_length]
        )
        for block in self.backbone.layers:
            if hasattr(block.attn, "causal_mask"):
                block.attn.causal_mask = block.attn.causal_mask[:, :, :context_length, :context_length]

            # update context length attribute in MultiHeadSelfAttention
            block.attn.context_length = context_length

    def _init_weights(self, module: nn.Module) -> None:
        normal_init_modules = (nn.Linear, nn.Embedding)
        if isinstance(module, normal_init_modules):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, in_indices: torch.LongTensor) -> torch.FloatTensor:
        device = in_indices.device

        B, T = in_indices.size()

        positions = torch.arange(0, T, dtype=torch.long, device=device)  # [T]
        token_embeddings = self.backbone.token_embeddings(in_indices)  # [B, T, D]
        positional_embeddings = self.backbone.position_embeddings(positions)  # [T, D]
        # fmt: off
        positional_embeddings = positional_embeddings.unsqueeze(0) # .expand(B, -1, -1) # [B, T, D]
        # fmt: on

        z = self.backbone.dropout(token_embeddings + positional_embeddings)  # [B, T, D]

        for block in self.backbone.layers:
            z = block(z)  # [B, T, D]

        z = self.backbone.ln_final(z)  # [B, T, D]

        logits = self.head(z)  # [B, T, V]
        return cast(torch.FloatTensor, logits)  # [B, T, V]


def initialize_model(
    config: GPTConfig,
    device: str = "cuda",
) -> GPT:
    if config.d_ff is None:
        config.d_ff = 4 * config.d_model

    model = GPT(config)
    return model.to(device)


def get_random_batch(
    batch_size: int,
    context_length: int,
    vocab_size: int,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.randint(  # [B, T]
        0,
        vocab_size,
        (batch_size, context_length),
        dtype=torch.long,
        device=device,
    )

    targets = torch.randint(  # [B, T]
        0,
        vocab_size,
        (batch_size, context_length),
        dtype=torch.long,
        device=device,
    )
    return inputs, targets

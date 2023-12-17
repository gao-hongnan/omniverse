from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from omnivault.transformer.modules.attention.base import Attention


class ScaledDotProductAttention(Attention):
    """
    Implements scaled dot-product attention mechanism.

    This class is a derived instance of the Attention class that computes the
    scaled dot-product attention, defined by the following operation:

    .. math::
        \\text{Attention}(Q, K, V) = \\text{softmax} \\left( \\frac{QK^T}{\\sqrt{d_k}} \\right) V

    where:
    - Q is the query matrix
    - K is the key matrix
    - V is the value matrix
    - d_k is the dimension of the keys

    The attention mechanism can be applied in two different contexts: self-attention
    and cross-attention. Self-attention allows the model to integrate information
    from the entire sequence, while cross-attention allows the model to focus on
    information from a different sequence (e.g., encoder outputs).

    Methods
    -------
    forward(query, key, value, mask)
        Computes the forward pass for the scaled dot-product attention.
    """

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.BoolTensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass for scaled dot-product attention.

        This function applies the attention mechanism on the input tensors `query`,
        `key`, and `value`. It's worth noting that for cross-attention, the sequence
        lengths of `query` and `key`/`value` may differ. This is because `query` is
        usually projected from the decoder's states, while `key` and `value` are from
        the encoder's states.

        Notations
        ---------
        - `B`  : Batch size
        - `D`  : Embedding dimension
        - `H`  : Number of heads
        - `d_k`: Dimension of the keys    = D // H
        - `d_q`: Dimension of the queries = D // H
        - `d_v`: Dimension of the values  = D // H
        - `N`  : Batch size
        - `T`  : Sequence length for `query`
        - `S`  : Sequence length for `key` and `value`
        - `L`  : Sequence length for `query`, `key` and `value` generic.

        NOTE: We use `L` in our notes instead of `T` and `S` since we assume all query,
        key and value are of same length.

        Also, we often denote the dimension of the keys and queries as `d_k`
        instead of `d_k` and `d_q` respectively because both must have
        the same dimensionality for them to be multiplied together.

        TODO: which shape is for cross-self?

        Parameters
        ----------
        query:  A tensor of query vectors representing the set of elements each sequence
                is seeking to attend to. It contains a batch of sequences, each with a set of
                vectors across multiple attention heads.
                    type :  torch.Tensor
                    shape: `(N, H, S or T, d_q)` where `d_q = D // H`
        key  :  A tensor of key vectors that are paired with values to form a mapping. The
                dot product of a query with these keys determines the attention weight for the
                corresponding values.
                    type :  torch.Tensor
                    shape: `(N, H, S or T, d_k)` where `d_k = D // H`
        value: A tensor of value vectors that are aggregated based on the attention
               weights to form the output of the attention mechanism.
                    type :  torch.Tensor
                    shape: `(N, H, S or T, d_v)` where `d_v = D // H`
        mask : An optional boolean mask tensor that can be used to mask out certain positions
               from the attention mechanism. For self-attention, the mask shape is typically
               `(B, T, T)`. For cross-attention, the mask typically has a shape of `(B, T, S)`
               allowing different target positions to attend to different source positions.
               Here, `T` is the sequence length of the queries (note `T` is the same for
               self-attention), `T_k` and `T_v` are the sequence lengths of the keys and values
               which could be equal to `T` in self-attention or vary in cross-attention, and
               `S` is the sequence length of the source (encoder) when using cross-attention.

               However, to cater to the head dimension `H`, right after the `B` dimension, we
               will need to add (unsqueeze) an dimension to have `(B, 1, T, T)` for
               self-attention and `(B, 1, T, S)` for cross-attention.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The context vectors and the attention weights. The context vectors are the weighted sum
            of the `value` vectors, representing the information to be attended to.
            The attention weights represent the attention probabilities.

            - Context Vectors shape:   `(N, T, d_k)`
            - Attention Weights shape: `(N, T, S)`
        """
        # fmt: off
        d_q               = query.size(dim=-1)

        attention_scores  = torch.matmul(query, key.transpose(dim0=-2, dim1=-1)) / torch.sqrt(torch.tensor(d_q).float())
        attention_scores  = attention_scores.masked_fill(mask == 0, float("-inf")) if mask is not None else attention_scores

        attention_weights = attention_scores.softmax(dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vector    = torch.matmul(attention_weights, value)
        # fmt: on
        return context_vector, attention_weights


class MultiHeadedAttention(nn.Module):
    __slots__ = [
        "d_model",
        "d_k",
        "d_q",
        "d_v",
        "H",
        "W_Q",
        "W_K",
        "W_V",
        "W_O",
        "attention",
        "dropout",
        "context_vector",
        "attention_weights",
    ]

    def __init__(
        self,
        attention: Attention,
        H: int,
        d_model: int,
        dropout: float = 0.1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert d_model % H == 0

        # fmt: off
        self.d_model   = d_model       # D
        self.d_k       = d_model // H  # stay true to notations
        self.d_q       = d_model // H
        self.d_v       = d_model // H

        self.H         = H             # number of heads

        # shadow my notations, actually they are of shape D x D.
        self.W_Q       = nn.Linear(self.d_model, self.d_q * self.H, bias=bias)  # D x D
        self.W_K       = nn.Linear(self.d_model, self.d_k * self.H, bias=bias)
        self.W_V       = nn.Linear(self.d_model, self.d_v * self.H, bias=bias)
        self.W_O       = nn.Linear(self.d_model, self.d_model, bias=bias)

        self.attention = attention
        self.dropout   = nn.Dropout(p=dropout, inplace=False)

        self.context_vector: torch.Tensor
        self.attention_weights: torch.Tensor

        # self._reset_parameters()
        # fmt: on

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        """
        Notations
        ---------
        B:      Batch size
        S or L: Source sequence length
        T or L: Target sequence length
        D:      Embedding dimension

        Parameters
        ----------
        query:  Although named as query, it is the embeddings Z from the token_embedding + positional_embedding layer.
                type:  torch.Tensor
                shape: (B, S or T, D)
        key:    Although named as key, it is the embeddings Z from the token_embedding + positional_embedding layer.
                type:  torch.Tensor
                shape: (B, S or T, D)
        value:  Although named as value, it is the embeddings Z from the token_embedding + positional_embedding layer.
                type:  torch.Tensor
                shape: (B, S or T, D)
        mask:   Mask to be applied to the attention scores.
                type:  torch.BoolTensor
                shape: (B, 1, S or T, S or T)

        Variables
        ---------
        W_Q.weight (D, D)

        """
        # fmt: off
        if mask is not None:
            assert mask.ndim     == 4, f"Mask should have 4 dimensions but got {mask.ndim}."
            assert mask.shape[0] == query.shape[0], ("Batch size of mask and query must match.")
            assert mask.shape[1] == 1, ("Mask should have shape (batch_size, 1, seq_len, seq_len).")
            assert mask.shape[2] == mask.shape[3] == query.shape[1], ("Mask should have shape (batch_size, 1, seq_len, seq_len).")


        Q = self.W_Q(query).contiguous() # Z @ W_Q -> LxD @ DxD = LxD
        K = self.W_K(key).contiguous()   # Z @ W_K
        V = self.W_V(value).contiguous() # Z @ W_V


        Q = self.transpose_qkv(Q)        # [B, H, L, D]
        K = self.transpose_qkv(K)
        V = self.transpose_qkv(V)

        # Attention
        # same as the other code: x = torch.matmul(p_atten, value)
        self.context_vector, self.attention_weights = self.attention(Q, K, V, mask)
        context_vector_concat                       = self.reverse_transpose_qkv(self.context_vector)
        # fmt: on

        # mypy complains because it infers `O` as `Any` but it is actually a tensor.
        # You can either cast it to tensor or use `self.W_O.forward(context_vector_concat)`.
        O = self.W_O(context_vector_concat)
        return O  # type: ignore[no-any-return]

    def _reset_parameters(self) -> None:
        """See PyTorch's code for inspiration!"""
        # we assume _qkv_same_embed_dim is True
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.xavier_uniform_(self.W_O.weight)

    def transpose_qkv(self, q_or_k_or_v: torch.Tensor) -> torch.Tensor:
        """Transposition for parallel computation of multiple attention heads.
        TODO: Why does transpose allow parallel computation?
        """
        # fmt: off
        # 1. q_or_k_or_v is shape (B, L, D)
        # 2. aim to make it of shape (B, L, H, D / H = d_qkv)
        batch_size, seq_len, _ = q_or_k_or_v.shape
        q_or_k_or_v            = q_or_k_or_v.view(batch_size, seq_len, self.H, self.d_model // self.H)

        # 3. switch H from 3rd to 2nd dimension, or in python swap 2nd to 1st
        q_or_k_or_v            = q_or_k_or_v.permute(0, 2, 1, 3)
        # fmt: on
        return q_or_k_or_v

    def reverse_transpose_qkv(self, q_or_k_or_v: torch.Tensor) -> torch.Tensor:
        """Reverse the transposition operation for concatenating multiple attention heads."""
        # fmt: off
        # 1. q_or_k_or_v is shape (B, H, L, D / H = d_qkv)
        # 2. aim to make it of shape (B, L, H, D / H = d_qkv)
        q_or_k_or_v = q_or_k_or_v.permute(0, 2, 1, 3)

        # 3. Merge H and d_qkv into D
        batch_size, seq_len, _, _ = q_or_k_or_v.shape
        q_or_k_or_v = q_or_k_or_v.contiguous().view(batch_size, seq_len, self.d_model)
        # fmt: on
        return q_or_k_or_v

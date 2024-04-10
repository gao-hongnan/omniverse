r"""
Scaled Dot-Product Attention Module
===================================

This module defines the Scaled Dot-Product Attention mechanism, which is a key
component in transformer architectures. It follows the equations:

.. math::
    \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax} \left( \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V}

The attention mechanism is computed as follows:

1. Find the embedding dimension $D$ or $d_q$ from the query feature vector. Note
   the query vector has the below representation:

   .. math::
       \mathbf{Q}   = \mathbf{Z} @ \mathbf{W}^{\mathbf{Q}} \in \mathbb{R}^{T \times D}
       \mathbf{Q}_h = \mathbf{Q} @ \mathbf{W}_{h}^{\mathbf{Q}} \in \mathbb{R}^{T \times d_q}
       d_q = \frac{D}{H}

    However, we note :math:\mathbf{Q}_h is more of a symbolic understanding
    than a real implementation.

2. Compute the dot product of the query feature vector :math:`\mathbf{Q}` with
   the transpose of the key feature vector :math:`\mathbf{K}`.
   Note since in practice the query, key and value are batched, we would have
   :math:`\mathbf{K} \in \mathbb{R}^{B \times T \times D}` and so we operate the
   transpose on the last two dimensions, specified by `dim0` and `dim1`.
   key.transpose(dim0=-2, dim1=-1) means let the second last dimension be the
   last dimension and let the last dimension be the second last dimension.

   This is the attention scores. Initially, the raw attention scores are scaled
   by the inverse square root of the dimensionality of the key vectors to
   stabilize the gradients during training.

3. Apply mask to the scores if mask is not None. Softmax of -INF is zero,
   effectively zeroing out masked logits. If False in mask is we ignore, then we
   should fill attention scores with -INF. masked_fill fills tensor with value
   -INF whenever mask's value is True. So we want the converse, so mask==0 means
   whenever mask has False, then mask==0 evaluates to True, and we fill that.

   Now attention scores are filled with -INF whenever mask is False.

4. Context vector formation: The final context vectors are calculated as a
weighted sum of the value vectors, with the weights provided by the
attention weights. This step effectively combines the value vectors based on
the computed attention, focusing the model's attention on certain elements.

    .. math::
        \text{context_vector} = \text{attention_weights} \cdot V

5. Output generation: The computed context vectors and the attention weights
are then output from the attention mechanism. The context vectors serve as a
synthesis of the input sequence as seen by the model, incorporating the most
relevant pieces of information according to the attention weights.

Self-Attention versus Cross-Attention
=====================================

In the context of attention mechanisms, especially in sequence-to-sequence
models like transformers, the terms "query", "key", and "value" can come from
different parts of the model and have different sequence lengths:

- The "query" usually comes from the decoder side of a transformer model. The
  sequence length of the query is often referred to as $T$ (target sequence
  length) because it relates to the target sequence that the model is trying to
  generate.
- The "key" and "value" usually come from the encoder side. The sequence length
  of the key and value is often referred to as $S$ (source sequence length)
  because they relate to the source sequence that is being encoded.

This distinction becomes particularly important in cross-attention modules where
the decoder (target) needs to attend to the output of the encoder (source). In
self-attention modules within either the encoder or decoder, the sequence
lengths of query, key, and value would be the same, and this distinction
wouldn't be necessary.

Multi-Head Attention Module
===========================

This module defines the Multi-Head Attention mechanism, which is a key
component in transformer architectures. It follows the equations:

.. math::
    \text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, ..., \text{head}_H)W^O

where each head is computed as:

.. math::
    \text{head}_h = \text{Attention}(QW^Q_h, KW^K_h, VW^V_h) where h \in [1, H]

Here, \mathbf{Q}, \mathbf{K}, \mathbf{V} are the query, key, and value vectors. W^Q_h, W^K_h, W^V_h are
parameter matrices. H is the number of heads.

Notations
=========

- For a decoder-only model (like GPT), where self-attention is used, the
  sequence lengths for query, key, and value are the same. It's common to use a
  single term for the sequence length, such as $L$ or $T$, because there is no
  need to differentiate between different sequence lengths within the attention
  mechanism.

- For a full encoder-decoder model (like the Transformer), it is beneficial to
  differentiate the sequence lengths. Here, $T$ typically denotes the sequence
  length of the query vectors that come from the decoder, and $S$ denotes the
  sequence length of the key and value vectors that come from the encoder. This
  distinction is important during cross-attention, where the decoder attends to
  the output of the encoder, and thus the lengths can be different.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from omnivault.transformer.modules.attention.base import Attention

__all__ = ["MultiHeadedAttention", "ScaledDotProductAttention"]


class ScaledDotProductAttention(Attention):
    r"""Implements scaled dot-product attention mechanism.

    This class is a derived instance of the `Attention` class that computes the
    scaled dot-product attention, defined by the following operation:

    .. math::
        \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) \mathbf{V}

    where:

    -   Q is the query matrix
    -   K is the key matrix
    -   V is the value matrix
    -   d_k is the dimension of the keys

    Note
    ----
    The attention mechanism can be applied in two different contexts: self-attention
    and cross-attention. Self-attention allows the model to integrate information
    from the entire sequence, while cross-attention allows the model to focus on
    information from a different sequence (e.g., encoder outputs).

    Consequently, we have the following implicit assumptions:

    -   For self-attention, the sequence lengths of `query`, `key`, and `value` are
        the same. This is because the query, key, and value are all derived from the
        same sequence `z` (e.g., the output of the token embedding and positional
        encoding layers).

    1. **Self-Attention** (also known as intra-attention) is used within a single
    sequence to relate different positions of this sequence. In self-attention,
    the queries (Q), keys (K), and values (V) all come from the same previous
    layer output. This is common in both the encoder and the decoder layers of a
    Transformer. For example, in a standalone GPT-Decoder variant, the initial
    queries, keys, and values all originates from the same input sequence, which
    becomes the `z` in the token embedding and positional encoding layers.

    2. **Cross-Attention** is used in the decoder to attend to the output of the
    encoder. In cross-attention, the queries (Q) come from the previous layer of
    the decoder, but the keys (K) and values (V) come from the output of the
    encoder. This allows each position in the decoder to attend to all positions
    in the input sequence.
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
        - B   : Batch size
        - D   : Embedding dimension
        - H   : Number of heads
        - d_k : Dimension of the keys    = D // H
        - d_q : Dimension of the queries = D // H
        - d_v : Dimension of the values  = D // H
        - T   : Sequence length for `query`
        - S   : Sequence length for `key` and `value`
        - L   : Sequence length for `query`, `key` and `value` generic.

        Note
        ----
        -   We use `L` in our notes instead of `T` and `S` since we assume all query,
            key and value are of same length as we are dealing with self-attention in
            GPT decoder.

        -   We often denote the dimension of the keys and queries as `d_k` instead of
            `d_k` and `d_q` respectively because both must have the same dimensionality
            for them to be multiplied together.

        Parameters
        ----------
        query:  A tensor of query vectors representing the set of elements each sequence
                is seeking to attend to. It contains a batch of sequences, each with a set of
                vectors across multiple attention heads.
                    type :  torch.Tensor
                    shape: `(B, H, T, d_q)` where `d_q = D // H`
                    shape: `(B, H, L, d_q)` if in pure self-attention (GPT)
        key  :  A tensor of key vectors that are paired with values to form a mapping. The
                dot product of a query with these keys determines the attention weight for the
                corresponding values.
                    type :  torch.Tensor
                    shape: `(B, H, S, d_k)` where `d_k = D // H`
                    shape: `(B, H, L, d_k)` if in pure self-attention (GPT)
        value: A tensor of value vectors that are aggregated based on the attention
               weights to form the output of the attention mechanism.
                    type :  torch.Tensor
                    shape: `(B, H, S, d_v)` where `d_v = D // H`
                    shape: `(B, H, L, d_v)` if in pure self-attention (GPT)
        mask : An optional boolean mask tensor that can be used to mask out certain positions
               from the attention mechanism. For self-attention, the mask shape is typically
               `(B, T, T)` or `(B, L, L). For cross-attention, the mask typically has a shape of
               `(B, T, S)` allowing different target positions to attend to different source positions.
               Here, `T` is the sequence length of the queries (note `T` is the same for
               self-attention), `S` is the sequence lengths of the keys and values
               which could be equal to `T` in self-attention or vary in cross-attention, and
               `S` is the sequence length of the source (encoder) when using cross-attention.

               However, to cater to the head dimension `H`, right after the `B` dimension, we
               will need to add (unsqueeze) an dimension to have `(B, 1, T, T)` for
               self-attention and `(B, 1, T, S)` for cross-attention.

        Returns
        -------
        context_vector, attention_weights: Tuple[torch.Tensor, torch.Tensor]
            The context vectors and the attention weights. The context vectors are the weighted sum
            of the `value` vectors, representing the information to be attended to.
            The attention weights represent the attention probabilities.

            - Context Vectors shape:   `(B, T, d_v)`
            - Attention Weights shape: `(B, T, S)` or `(B, H, T, S)` or `(B, H, L, L)` if in pure self-attention (GPT)
        """
        # fmt: off
        d_q               = query.size(dim=-1)

        attention_scores  = torch.matmul(query, key.transpose(dim0=-2, dim1=-1)) / torch.sqrt(torch.tensor(d_q).float())        # [B, H, T, d_q] @ [B, H, d_q, T] = [B, H, T, T]
        attention_scores  = attention_scores.masked_fill(mask == 0, float("-inf")) if mask is not None else attention_scores    # [B, H, T, T]

        attention_weights = attention_scores.softmax(dim=-1)        # [B, H, T, T]
        attention_weights = self.dropout(attention_weights)         # [B, H, T, T]

        context_vector    = torch.matmul(attention_weights, value)  # [B, H, T, T] @ [B, H, T, d_v] = [B, H, T, d_v]
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
        assert d_model % H == 0, "The number of heads must divide the embedding dimension."

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

        # self._init_weights()
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
        H:      Number of heads

        Parameters
        ----------
        query:  Although named as query, it is the embeddings `z` from the token_embedding + positional_embedding layer.
                type:  torch.Tensor
                shape: (B, S or T, D)
        key:    Although named as key, it is the embeddings `z` from the token_embedding + positional_embedding layer.
                type:  torch.Tensor
                shape: (B, S or T, D)
        value:  Although named as value, it is the embeddings `z` from the token_embedding + positional_embedding layer.
                type:  torch.Tensor
                shape: (B, S or T, D)
        mask:   Mask to be applied to the attention scores.
                type:  torch.BoolTensor
                shape: (B, 1, S or T, S or T)

        Returns
        -------
        O:  The output of the multi-headed attention mechanism.
            type:  torch.Tensor
            shape: (B, S or T, D)

        Variables
        ---------
        W_Q.weight (D, D)
        W_K.weight (D, D)
        W_V.weight (D, D)
        W_O.weight (D, D)
        """
        # fmt: off
        if mask is not None:
            assert mask.ndim     == 4, f"Mask should have 4 dimensions but got {mask.ndim}."
            assert mask.shape[0] == query.shape[0], ("Batch size of mask and query must match.")
            assert mask.shape[1] == 1, ("Mask should have shape (batch_size, 1, seq_len, seq_len).")
            assert mask.shape[2] == mask.shape[3] == query.shape[1], ("Mask should have shape (batch_size, 1, seq_len, seq_len).")


        Q = self.W_Q(query).contiguous() # Z @ W_Q -> BxTxD @ DxD = BxTxD
        K = self.W_K(key).contiguous()   # Z @ W_K
        V = self.W_V(value).contiguous() # Z @ W_V

        Q = self.transpose_qkv(Q)        # splitting happens -> [B, H, L, D]
        K = self.transpose_qkv(K)
        V = self.transpose_qkv(V)

        # Attention
        self.context_vector, self.attention_weights = self.attention(Q, K, V, mask)
        context_vector_concat                       = self.reverse_transpose_qkv(self.context_vector)
        # fmt: on

        # mypy complains because it infers `O` as `Any` but it is actually a tensor.
        # You can either cast it to tensor or use `self.W_O.forward(context_vector_concat)`.
        O = self.W_O(context_vector_concat)  # context_vector_concat @ W_O -> LxD @ DxD = LxD
        return O  # type: ignore[no-any-return]

    def _init_weights(self) -> None:
        """See PyTorch's MultiHeadAttention code for reference."""
        # we assume _qkv_same_embed_dim is True
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.xavier_uniform_(self.W_O.weight)

    def transpose_qkv(self, q_or_k_or_v: torch.Tensor) -> torch.Tensor:
        """Transposition for parallel computation of multiple attention heads.
        Why does transpose allow parallel computation? So originally the shape of
        the query, key, and value is (B, L, D), and we want to split the D into H
        heads to become (B, L, H, D / H). But this is not the shape we want (could
        be due to efficiency reasons), so we transpose the shape to (B, H, L, D / H)
        so all heads can be computed in parallel (efficiently).

        Parameters
        ----------
        q_or_k_or_v: The query, key, or value tensor.
            type:  torch.Tensor
            shape: (B, L, D)

        Returns
        -------
        q_or_k_or_v: The transposed query, key, or value tensor.
            type:  torch.Tensor
            shape: (B, H, L, D / H)
        """
        # fmt: off
        # 1. q_or_k_or_v is shape (B, L, D)
        # 2. aim to make it of shape (B, L, H, D / H = d_qkv)
        batch_size, seq_len, _ = q_or_k_or_v.shape
        q_or_k_or_v            = q_or_k_or_v.view(batch_size, seq_len, self.H, self.d_model // self.H)

        # 3. switch H from 3rd to 2nd dimension, or in python swap 2nd to 1st dimension and 1st to 2nd dimension
        #    shape (B, H, L, D / H = d_qkv)
        q_or_k_or_v            = q_or_k_or_v.permute(0, 2, 1, 3)
        # fmt: on
        return q_or_k_or_v

    def reverse_transpose_qkv(self, q_or_k_or_v: torch.Tensor) -> torch.Tensor:
        """Reverse the transposition operation for concatenating multiple attention heads.

        Parameters
        ----------
        q_or_k_or_v: The query, key, or value tensor.
            type:  torch.Tensor
            shape: (B, H, L, D / H)

        Returns
        -------
        q_or_k_or_v: The transposed query, key, or value tensor.
            type:  torch.Tensor
            shape: (B, L, D)
        """
        # fmt: off
        # 1. q_or_k_or_v is shape (B, H, L, D / H = d_qkv)
        # 2. aim to make it of shape (B, L, H, D / H = d_qkv)
        q_or_k_or_v = q_or_k_or_v.permute(0, 2, 1, 3)

        # 3. Merge H and d_qkv into D
        batch_size, seq_len, _, _ = q_or_k_or_v.shape
        q_or_k_or_v = q_or_k_or_v.contiguous().view(batch_size, seq_len, self.d_model)
        # fmt: on
        return q_or_k_or_v

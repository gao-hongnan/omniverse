r"""
Scaled Dot-Product Attention Module
===================================

This module defines the Scaled Dot-Product Attention mechanism, which is a key
component in transformer architectures. It follows the equations:

.. math::
    \text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V

The attention mechanism is computed as follows:
1. Find the embedding dimension $D$ or $d_q$ from the query feature vector. Note
   the query vector has the below representation:

   .. math::
       Q   = Z @ W_Q  \in  \mathbb{R}^{L \times D}
       Q_h = Q @ W_Q^h \in \mathbb{R}^{L \times d_q}

2. Compute the dot product of the query feature vector with the key feature
   vector. Note since key is of dim (batch_size, L, $d_k$) so we operate the
   transpose on the last two dimensions, specified by dim0 and dim1.
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
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from torch import nn


# FIXME: consider removing ABC since nn.Module ensures forward method to be implemented.
class Attention(ABC, nn.Module):
    """
    Base class for attention mechanisms.

    This abstract class provides a scaffold for attention mechanisms, with a
    dropout layer for regularization included. Subclasses are expected to
    implement the `forward` method.

    Attributes
    ----------
    dropout : The dropout layer applied to the attention scores.
        type: nn.Dropout

    Note
    ----
    ABC method might be redundant since inheritance from nn.Module ensures
    forward method to be implemented.
    """

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=False)

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass for the attention mechanism."""
        raise NotImplementedError("The forward method must be implemented by the subclass.")

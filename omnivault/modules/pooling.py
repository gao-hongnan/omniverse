from __future__ import annotations

from typing import overload

import torch
from torch import nn

__all__ = ["LastTokenPooling"]


class LastTokenPooling(nn.Module):
    """Last token pooling layer - specifically for decoder only models to do
    fine-tuning on sequence classification tasks."""

    def __init__(self, pre_head_pooling: bool = True) -> None:
        super().__init__()
        self.pre_head_pooling = pre_head_pooling

    @overload
    def forward(self, last_hidden_state: torch.Tensor, logits: None = None) -> torch.Tensor: ...

    @overload
    def forward(self, last_hidden_state: None, logits: torch.Tensor) -> torch.Tensor: ...

    def forward(
        self, last_hidden_state: torch.Tensor | None = None, logits: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass for the pooling layer.

        Parameters
        ----------
        last_hidden_state:  Hidden state of the last layer.
                            type:  torch.Tensor
                            shape: (B, T, D)
        logits:             Logits from the last layer.
                            type:  torch.Tensor
                            shape: (B, T, C)

        Notes
        -----
        In both cases, we will slice the `T` dimension to get the last token's
        hidden state or logits. For example, if `last_hidden_state` is provided,
        then we have `[B, T, D] -> [B, D]` and if `logits` is provided, then we
        have `[B, T, C] -> [B, C]`.
        """
        if self.pre_head_pooling:
            assert last_hidden_state is not None, "last_hidden_state must be provided when pre_head is True"
            pooled_hidden_state = last_hidden_state[:, -1, :]
            return pooled_hidden_state
        else:
            assert logits is not None, "logits must be provided when pre_head is False"
            pooled_logits = logits[:, -1, :]
            return pooled_logits


class MeanPooler(nn.Module):
    """
    # Qwen/Qwen1.5-0.5B
    padding side = right
    B=2, T=3, D=4
    attention_mask: [B, T] -> [[1, 1, 0], [1, 0, 0]]
    last_hidden_state: [B, T, D] -> [
                                        [[1, 2, 3, 4],    [5, 6, 7, 8],     [1, 1, 5, 2]],
                                        [[9, 10, 11, 12], [13, 14, 15, 16], [1, 3, 2, 2]]
                                    ]
    input_mask_expanded: [B, T, D] ->   [
                                            [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]],
                                            [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]
                                        ]

    sum_embeddings: [B, D] -> the idea is simple, you want the sequence position
    for which the attention mask is 1, and sum the embeddings for that position.
    In other words, if the attention mask is 0, you want to nullify the embedding
    for that position. This is achieved by multiplying the embeddings with the
    attention mask. This is done for all the positions in the sequence. This
    effectively make [1,1,5,2] * [0,0,0,0] = [0,0,0,0] in the example above.
    We just want:

    1st sequence in the batch to become shape [D] by:
        - do a multiplication of the last hidden state with the attention mask
            [1, 2, 3, 4] * [1, 1, 1, 1] = [1, 2, 3, 4]
            [5, 6, 7, 8] * [1, 1, 1, 1] = [5, 6, 7, 8]
            [1, 1, 5, 2] * [0, 0, 0, 0] = [0, 0, 0, 0]

            leads to stacked shape of [T, D] for the first sequence

        - sum the embeddings for each position in the sequence
            [1, 2, 3, 4] + [5, 6, 7, 8] + [0, 0, 0, 0] = [6, 8, 10, 12]

                leads to shape [D] for the first sequence
        - divide the sum by the sum of the attention mask, in this example
            our sum of the attention mask is [1, 1, 1, 1] + [1, 1, 1, 1] + [0, 0, 0, 0] = [2, 2, 2, 2]
            in other words we have 2 valid tokens in the sequence to be divided
    """

    def forward(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Execute the forward pass of the MeanPooler.

        Parameters
        ----------
        last_hidden_state : torch.Tensor
            The last hidden state of the model.
            Shape: [B, T, D]
        attention_mask : torch.Tensor
            The attention mask of the model.
            Shape: [B, T]
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class AttentionHead(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.W = nn.Linear(in_features, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.out_features = hidden_dim

    def forward(self, features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        weights_mask = attention_mask.unsqueeze(-1)
        att = torch.tanh(self.W(features))
        score = self.V(att)
        score[attention_mask == 0] = -1e4
        attention_weights = torch.softmax(score, dim=1)
        context_vector = torch.sum(attention_weights * weights_mask * features, dim=1)
        return context_vector

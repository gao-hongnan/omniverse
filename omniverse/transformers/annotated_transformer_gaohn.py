import copy
import math
from abc import ABC, abstractmethod

import rich
import torch
from attention import attention
from d2l import torch as d2l
from rich.pretty import pprint
from torch import nn
from torch.nn import Transformer

from common_utils.core.common import seed_all

seed_all(42, seed_torch=True)


class MultiHeadedAttention(nn.Module):
    def __init__(self, H, d_model, dropout=0.1, bias=False) -> None:
        "Take in model size and number of heads."
        super().__init__()
        assert d_model % H == 0

        self.d_model = d_model  # D
        self.d_k = d_model // H  # stay true to notations
        self.d_q = d_model // H
        self.d_v = d_model // H

        self.H = H  # number of heads

        # shadow my notations
        self.W_q = nn.Linear(self.d_model, self.d_q * self.H, bias=bias)  # D x D
        self.W_k = nn.Linear(self.d_model, self.d_k * self.H, bias=bias)
        self.W_v = nn.Linear(self.d_model, self.d_v * self.H, bias=bias)
        self.W_o = nn.Linear(self.d_model, self.d_model, bias=bias)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, embeddings, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches, seq_len, _ = embeddings.size()

        # Apply linear transformations to compute Q, K, V
        # NOTE: here is an important misconception that if you have
        # 8 heads, then you SPLIT the embeddings into 8 parts and
        # then apply linear transformations to each part. This is
        # WRONG. You apply linear transformations to the whole
        # embeddings and then split the result into 8 parts.
        W_q = self.W_q.weight  # D x D
        W_k = self.W_k.weight  # D x D
        W_v = self.W_v.weight  # D x D

        # NOTE: in pytorch, you need to transpose the weight matrix if
        # you see their formula, so this has a bit of a different
        # notation than in our notes. In our notes, we do not need to transpose.

        Q = embeddings @ W_q.T  # Z @ W_q
        K = embeddings @ W_k.T  # Z @ W_k
        V = embeddings @ W_v.T  # Z @ W_v
        assert tensors_are_same(Q, self.W_q(embeddings))
        assert tensors_are_same(K, self.W_k(embeddings))
        assert tensors_are_same(V, self.W_v(embeddings))

        # Q = self.W_q(embeddings) # Z @ W_q
        # K = self.W_k(embeddings) # Z @ W_k
        # V = self.W_v(embeddings) # Z @ W_v
        assert Q.shape == (nbatches, seq_len, self.d_q * self.H)

        # Splitting into multiple heads
        Q_heads = []
        K_heads = []
        V_heads = []

        for head in range(self.H):
            # ASSUMING d_q == d_k == d_v
            # NOTE: see my notes on confusion of paper's usage of
            # W^{q}_i, W^{k}_i, W^{v}_i where in fact the
            # weights are shared across heads via W^{q}, W^{k}, W^{v}
            head_start = head * self.d_q
            head_end = (head + 1) * self.d_q

            # NOTE: W_q_h, W_k_h, W_v_h are computed just to check that
            # Q_h = embeddings @ W_q_h^T
            W_q_h = W_q.T[:, head_start:head_end]  # D x d_q
            W_k_h = W_k.T[:, head_start:head_end]  # D x d_k
            W_v_h = W_v.T[:, head_start:head_end]  # D x d_v

            Q_h = Q[:, :, head_start:head_end]
            pprint(embeddings.shape)
            pprint(W_q_h.T.shape)
            assert tensors_are_same(Q_h, embeddings @ W_q_h)  # Z @ W^{q}_h

            K_h = K[:, :, head_start:head_end]
            assert tensors_are_same(K_h, embeddings @ W_k_h)

            V_h = V[:, :, head_start:head_end]
            assert tensors_are_same(V_h, embeddings @ W_v_h)

            assert Q_h.shape == (nbatches, seq_len, self.d_q)
            assert K_h.shape == (nbatches, seq_len, self.d_k)
            assert V_h.shape == (nbatches, seq_len, self.d_v)

            Q_heads.append(Q_h)
            K_heads.append(K_h)
            V_heads.append(V_h)

        # as of now we are at the stage
        # right before head_h = attention(Q_h, K_h, V_h)
        # so next step apply attention to each head h.

        # Apply attention to each head
        head_outputs = []
        for Q_h, K_h, V_h in zip(Q_heads, K_heads, V_heads):
            # apply Q,K,V to attention
            # x.shape = [nbatches, seq_len, d_v] = [2, 4, 100] or [2, 4, 50] if 2 heads
            x, attn = attention(Q_h, K_h, V_h, mask=mask, dropout=self.dropout)
            head_outputs.append(x)
            # FIXME: why is attn unused?
            self.attn = attn  # Store attention

        # Concatenate heads
        # NOTE: this is the step where we concatenate the heads
        # MultiHead(Q, K, V) = Concat(head_1, ..., head_H)W^{o}
        x_concat = torch.cat(head_outputs, dim=-1)
        pprint(x_concat.shape)
        assert x_concat.shape == (nbatches, seq_len, self.d_model)

        # Apply final linear transformation
        return self.W_o(x_concat)


def tensors_are_same(tensor1, tensor2, atol=1e-8, rtol=1e-5):
    return torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol)


if __name__ == "__main__":
    num_hiddens, num_heads = 100, 2
    model = MultiHeadedAttention(H=num_heads, d_model=num_hiddens, bias=False)
    batch_size, num_queries, num_kvpairs = 2, 4, 6
    valid_lens = torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))  # shape = [2, 4, 100]
    # Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
    # check the output shape
    # out_mask = model(X, Y, Y, mask=torch.ones((batch_size, num_queries, num_kvpairs)))
    out_no_mask = model.forward(embeddings=X, mask=None)
    # torch.save(out_no_mask, "ground_truth_attention.pt")
    # pprint(out_no_mask)
    # print(out_no_mask.shape)

    if num_heads == 1:
        loaded_out = torch.load("single_head_ground_truth_attention.pt")
    elif num_heads == 2:
        loaded_out = torch.load("double_head_ground_truth_attention.pt")
    else:
        raise ValueError("num_heads must be 1 or 2")

    if tensors_are_same(out_no_mask, loaded_out, atol=0, rtol=0):
        print("The tensors are the same!")
    else:
        print("The tensors are different!")

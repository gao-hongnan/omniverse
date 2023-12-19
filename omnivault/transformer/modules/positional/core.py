# import unittest
# from abc import ABC, abstractmethod

# import torch
# from d2l import torch as d2l
# from src.utils.reproducibility import seed_all
# from torch import nn


# class PositionalEncoding(ABC, nn.Module):
#     def __init__(self, d_model: int, dropout: float = 0.0) -> None:
#         super().__init__()
#         self.d_model = d_model
#         self.dropout = nn.Dropout(p=dropout, inplace=False)

#     @abstractmethod
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         ...


# class Sinusoid(PositionalEncoding):
#     def __init__(self, d_model: int, dropout: float = 0.0, max_seq_len: int = 3) -> None:
#         super().__init__(d_model, dropout)
#         self.max_seq_len = max_seq_len
#         self.d_model = d_model

#         P = self._init_positional_encoding()
#         self.register_buffer("P", P, persistent=True)

#     def _init_positional_encoding(self) -> torch.Tensor:
#         """Initialize the positional encoding tensor."""
#         P = torch.zeros((1, self.max_seq_len, self.d_model))
#         position = self._get_position_vector()
#         div_term = self._get_div_term_vector()
#         P[:, :, 0::2] = torch.sin(position / div_term)
#         P[:, :, 1::2] = torch.cos(position / div_term)
#         return P

#     def _get_position_vector(self) -> torch.Tensor:
#         """Return a vector representing the position of each token in a sequence."""
#         return torch.arange(self.max_seq_len, dtype=torch.float32).reshape(-1, 1)

#     def _get_div_term_vector(self) -> torch.Tensor:
#         """Return a vector representing the divisor term for positional encoding."""
#         return torch.pow(
#             10000,
#             torch.arange(0, self.d_model, 2, dtype=torch.float32) / self.d_model,
#         )

#     def forward(self, Z: torch.Tensor) -> torch.Tensor:
#         Z = self._add_positional_encoding(Z)
#         return self.dropout(Z)

#     def _add_positional_encoding(self, Z: torch.Tensor) -> torch.Tensor:
#         """Add the positional encoding tensor to the input tensor."""
#         return Z + self.P[:, : Z.shape[1], :].to(Z.device)


# class TestPositionalEncoding(unittest.TestCase):
#     def setUp(self) -> None:
#         seed_all(42, seed_torch=True)

#         # Initialize queries, keys, and values
#         # fmt: off
#         self.batch_size    = 1  # B
#         self.num_heads     = 2  # H
#         self.seq_len       = 60 # L
#         self.d_model = 32 # D
#         self.dropout       = 0.0
#         self.max_seq_len       = 1000

#         self.embeddings    = torch.zeros(self.batch_size, self.seq_len, self.d_model) # Z

#         self.pos_encoding  = Sinusoid(d_model=self.d_model, dropout=self.dropout, max_seq_len=self.max_seq_len)
#         # fmt: on

#         # Initialize the attention models
#         self.pos_encoding_d2l = d2l.PositionalEncoding(self.d_model, dropout=self.dropout, max_seq_len=self.max_seq_len)
#         self.pos_encoding_d2l.eval()

#     def test_positional_encoding_with_d2l_as_sanity_check(self) -> None:
#         # fmt: off
#         # d2l implementation
#         Z_d2l = self.pos_encoding_d2l(self.embeddings)
#         P_d2l = self.pos_encoding_d2l.P[:, : Z_d2l.shape[1], :]

#         # own implementation
#         Z     = self.pos_encoding(self.embeddings)
#         P     = self.pos_encoding.P[:, : Z.shape[1], :]
#         # fmt: on

#         # Test if both are close
#         self.assertTrue(torch.allclose(Z, Z_d2l))
#         self.assertTrue(torch.allclose(P, P_d2l))


# if __name__ == "__main__":
#     unittest.main()

from __future__ import annotations

from typing import List

import torch

from omnivault.transformer.core.vocabulary import Vocabulary


# TODO: vocabulary can be generic too.
class Tokenizer:
    def __init__(self, vocabulary: Vocabulary):
        self.vocabulary = vocabulary

    # NOTE: for simplicity, I do not add tokenizer, encode and decode here in fear of violating
    # the Liskov Substitution Principle, which is not something I want to worry
    # about in a naive implementation.


class TextCharacterTokenizer(Tokenizer):
    """
    A tokenizer for character-level text processing, responsible for tokenizing,
    encoding, and decoding text sequences using a given character vocabulary.
    """

    def tokenize(self, sequence: str) -> List[str]:
        return list(sequence)  # Tokenizes the text into a list of characters

    def encode(self, sequence: str) -> List[int]:
        return [self.vocabulary.token_to_index.get(char, -1) for char in sequence]  # -1 for unknown characters

    def decode(self, encoded_sequence: List[int] | torch.Tensor) -> str:
        if isinstance(encoded_sequence, torch.Tensor):
            encoded_sequence = encoded_sequence.tolist()
        return "".join([self.vocabulary.index_to_token.get(char, "") for char in encoded_sequence])

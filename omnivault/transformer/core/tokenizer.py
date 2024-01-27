from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar, Union

import torch

from omnivault.transformer.core.vocabulary import AdderVocabulary, TextCharacterVocabulary, Vocabulary

Vocabulary_t = TypeVar("Vocabulary_t", bound=Vocabulary)


class Tokenizer(ABC, Generic[Vocabulary_t]):
    def __init__(self, vocabulary: Vocabulary):
        self.vocabulary = vocabulary

    # NOTE: for simplicity, I do not add more args in tokenizer, encode and decode
    # here in fear of violating the Liskov Substitution Principle, which is not
    # something I want to worry about in a naive implementation.

    @abstractmethod
    def tokenize(self, sequence: str, add_special_tokens: bool = True) -> List[str]:
        """
        Tokenizes a sequence into a sequence (list) of tokens.

        Example
        -------
        >>> tokenizer = Tokenizer()
        >>> tokenizer.tokenize("The quick brown fox jumps over the lazy dog.")
        ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "."]

        Parameters
        ----------
        sequence : str
            The sequence to tokenize.
        add_special_tokens : bool
            Whether to add special tokens to the sequence of tokens, by default True.

        Returns
        -------
        List[str]
            The sequence of tokens.
        """

    @abstractmethod
    def encode(self, sequence: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encodes a sequence to its corresponding integer index.

        Parameters
        ----------
        sequence : str
            The sequence to encode.

        Returns
        -------
        List[int]
            The integer index corresponding to the token.
        """

    @abstractmethod
    def decode(self, encoded_sequence: List[int] | torch.Tensor, remove_special_tokens: bool = True) -> str:
        """
        Decodes an integer index back to its corresponding token.

        Parameters
        ----------
        index : int
            The integer index to decode.

        Returns
        -------
        str
            The token corresponding to the integer index.
        """


class AdderTokenizer(Tokenizer[AdderVocabulary]):
    def __init__(self, vocabulary: AdderVocabulary) -> None:
        super().__init__(vocabulary)

    def tokenize(self, sequence: str, add_special_tokens: bool = True) -> List[str]:
        tokens = [char for char in sequence]  # noqa: C416
        if add_special_tokens:
            tokens = [AdderVocabulary.BOS] + tokens + [AdderVocabulary.EOS]
        return tokens

    def encode(self, sequence: str, add_special_tokens: bool = True) -> List[int]:
        tokens: List[str] = self.tokenize(sequence, add_special_tokens=add_special_tokens)
        encoded_sequence: List[int] = [
            self.vocabulary.token_to_index.get(token, self.vocabulary.token_to_index[AdderVocabulary.UNK])
            for token in tokens
        ]
        return encoded_sequence

    def encode_batch(self, sequences: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        return [self.encode(sequence, add_special_tokens=add_special_tokens) for sequence in sequences]

    def decode(self, encoded_sequence: List[int] | torch.Tensor, remove_special_tokens: bool = True) -> str:
        if isinstance(encoded_sequence, torch.Tensor):
            encoded_sequence = encoded_sequence.tolist()
        decoded = "".join([self.vocabulary.index_to_token.get(char, AdderVocabulary.UNK) for char in encoded_sequence])

        if remove_special_tokens:
            decoded = re.sub(
                f"{AdderVocabulary.BOS}|{AdderVocabulary.EOS}|{AdderVocabulary.PAD}|{AdderVocabulary.UNK}",
                "",
                decoded,
            )
        return decoded

    def decode_batch(
        self, encoded_sequences: List[List[int]] | torch.Tensor, remove_special_tokens: bool = True
    ) -> List[str]:
        return [
            self.decode(encoded_sequence, remove_special_tokens=remove_special_tokens)
            for encoded_sequence in encoded_sequences
        ]


class TextCharacterTokenizer(Tokenizer[TextCharacterVocabulary]):
    """
    A tokenizer for character-level text processing, responsible for tokenizing,
    encoding, and decoding text sequences using a given character vocabulary.
    """

    def tokenize(self, sequence: str, add_special_tokens: bool = False) -> List[str]:
        tokens = list(sequence)  # Tokenizes the text into a list of characters
        if add_special_tokens:
            tokens = [TextCharacterVocabulary.BOS] + tokens + [TextCharacterVocabulary.EOS]
        return tokens

    def encode(self, sequence: str, add_special_tokens: bool = False) -> List[int]:
        tokens = self.tokenize(sequence, add_special_tokens=add_special_tokens)
        return [self.vocabulary.token_to_index.get(char, -1) for char in tokens]  # -1 for unknown characters

    def decode(self, encoded_sequence: List[int] | torch.Tensor, remove_special_tokens: bool = False) -> str:
        if isinstance(encoded_sequence, torch.Tensor):
            encoded_sequence = encoded_sequence.tolist()

        decoded_sequence = "".join([self.vocabulary.index_to_token.get(char, "") for char in encoded_sequence])

        if remove_special_tokens:
            # Remove special tokens from the decoded string
            special_tokens = {self.vocabulary.BOS, self.vocabulary.EOS, self.vocabulary.PAD, self.vocabulary.UNK}
            decoded_sequence = "".join(char for char in decoded_sequence if char not in special_tokens)

        return decoded_sequence


Tokenizers = Union[AdderTokenizer, TextCharacterTokenizer]

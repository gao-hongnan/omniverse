from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Dict, List, Type

from rich.pretty import pprint
from torch.utils.data import Dataset

from omnivault._types._alias import Token
from omnivault.transformers.config.constants import TOKENS


class Vocabulary(ABC):
    # Special tokens as class attributes
    BOS = "<BOS>"
    EOS = "<EOS>"
    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self, token_to_index: Dict[str, int], index_to_token: Dict[int, str]):
        self.token_to_index = token_to_index
        self.index_to_token = index_to_token

    @abstractmethod
    def tokenize(self, sentence: str, add_special_tokens: bool = True) -> List[str]:
        """
        Tokenizes a sentence into a sequence (list) of tokens.

        Example
        -------
        >>> tokenizer = Tokenizer()
        >>> tokenizer.tokenize("The quick brown fox jumps over the lazy dog.")
        ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "."]

        Parameters
        ----------
        sentence : str
            The sentence to tokenize.
        add_special_tokens : bool, optional
            Whether to add special tokens to the sequence of tokens, by default True.

        Returns
        -------
        List[str]
            The sequence of tokens.
        """

    @abstractmethod
    def encode(self, sentence: str) -> int:
        """
        Encodes a sentence to its corresponding integer index.

        Parameters
        ----------
        sentence : str
            The sentence to encode.

        Returns
        -------
        int
            The integer index corresponding to the token.
        """

    @abstractmethod
    def decode(self, sentence: str, remove_special_tokens: bool = True) -> str:
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

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of tokens in the vocabulary.

        Returns
        -------
        int
            The number of tokens in the vocabulary.
        """

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """
        Returns the size of the vocabulary.

        Returns
        -------
        int
            The size of the vocabulary.
        """

    @classmethod
    def from_tokens(cls: Type[Vocabulary], tokens: List[str]) -> Vocabulary:
        token_to_index = {token: idx for idx, token in enumerate(tokens)}
        index_to_token = {idx: token for token, idx in token_to_index.items()}
        return cls(token_to_index, index_to_token)


class AdderVocabulary(Vocabulary):
    # special tokens
    BOS = "<BOS>"
    EOS = "<EOS>"
    PAD = "<PAD>"
    UNK = "<UNK>"

    def tokenize(self, sentence: str, add_special_tokens: bool = True) -> List[str]:
        tokens = [char for char in sentence]
        if add_special_tokens:
            tokens = [AdderVocabulary.BOS] + tokens + [AdderVocabulary.EOS]
        return tokens

    def encode(self, sentence: str, add_special_tokens: bool = True) -> List[int]:
        tokens: List[str] = self.tokenize(
            sentence, add_special_tokens=add_special_tokens
        )
        return [
            self.token_to_index.get(token, self.token_to_index[AdderVocabulary.UNK])
            for token in tokens
        ]

    def encode_batch(
        self, sentences: List[str], add_special_tokens: bool = True
    ) -> List[List[int]]:
        return [
            self.encode(sentence, add_special_tokens=add_special_tokens)
            for sentence in sentences
        ]

    def decode(self, sentence: str, remove_special_tokens: bool = True) -> str:
        decoded = "".join(
            [self.index_to_token.get(char, AdderVocabulary.UNK) for char in sentence]
        )

        if remove_special_tokens:
            decoded = re.sub(
                f"{AdderVocabulary.BOS}|{AdderVocabulary.EOS}|{AdderVocabulary.PAD}|{AdderVocabulary.UNK}",
                "",
                decoded,
            )
        return decoded

    def decode_batch(
        self, sentences: List[List[int]], remove_special_tokens: bool = True
    ) -> List[str]:
        return [
            self.decode(sentence, remove_special_tokens=remove_special_tokens)
            for sentence in sentences
        ]

    def __len__(self) -> int:
        return len(self.token_to_index)

    @property
    def vocab_size(self) -> int:
        return len(self)


if __name__ == "__main__":
    vocab = AdderVocabulary.from_tokens(tokens=TOKENS)

    pprint(vocab.token_to_index)
    pprint(vocab.index_to_token)

    pprint(vocab.encode("1"))
    pprint(vocab.encode("+"))

    sentence = "15+57=072"
    sentences = ["15+57=072", "92+00=092", "95+53=148", "15+10=025"]

    encoded_sentence = vocab.encode(sentence)
    pprint(encoded_sentence)
    decoded_sentence = vocab.decode(encoded_sentence)
    pprint(decoded_sentence)

    encoded_sentences = vocab.encode_batch(sentences)
    pprint(encoded_sentences)
    decoded_sentences = vocab.decode_batch(encoded_sentences)
    pprint(decoded_sentences)

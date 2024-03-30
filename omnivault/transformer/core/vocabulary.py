from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Dict, List, Type, Union

import requests
from typing_extensions import override


class Vocabulary(ABC):
    """Base class for all vocabulary classes."""

    # Special tokens as class attributes
    BOS = "<BOS>"
    EOS = "<EOS>"
    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(
        self,
        token_to_index: Dict[str, int],
        index_to_token: Dict[int, str],
    ) -> None:
        self.token_to_index = token_to_index
        self.index_to_token = index_to_token

    def __len__(self) -> int:
        """
        Returns the total number of unique tokens in the vocabulary.

        Returns
        -------
        int
            The number of tokens in the vocabulary.
        """
        return len(set(self.token_to_index.keys()))

    @property
    def vocab_size(self) -> int:
        """
        Returns the size of the vocabulary.

        Returns
        -------
        int
            The size of the vocabulary.
        """
        return len(self)

    @classmethod
    def from_tokens(cls: Type[Vocabulary], tokens: List[str]) -> Vocabulary:
        token_to_index = {token: idx for idx, token in enumerate(tokens)}
        index_to_token = {idx: token for token, idx in token_to_index.items()}
        return cls(token_to_index, index_to_token)


class AdderVocabulary(Vocabulary):
    # special tokens extended from Vocabulary
    ADD = "+"
    EQUAL = "="
    TOKENS = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "+",
        "*",
        "-",
        "=",
        "<BOS>",
        "<EOS>",
        "<PAD>",
        "<UNK>",
    ]

    def __init__(self, token_to_index: Dict[str, int], index_to_token: Dict[int, str], num_digits: int) -> None:
        super().__init__(token_to_index, index_to_token)
        self.num_digits = num_digits

    @override
    @classmethod  # NOTE: why does overriding a classmethod with different signature not violate LSP?
    def from_tokens(cls: Type[AdderVocabulary], tokens: List[str], num_digits: int = 2) -> AdderVocabulary:
        token_to_index = {token: idx for idx, token in enumerate(tokens)}
        index_to_token = {idx: token for token, idx in token_to_index.items()}
        return cls(token_to_index, index_to_token, num_digits)


class TextCharacterVocabulary(Vocabulary):
    """
    A vocabulary class for character-level text processing. This class is designed
    to handle the encoding and decoding of characters in text data. It is particularly
    useful for tasks involving character-level models, such as GPT-style language models.

    The vocabulary consists of a set of unique characters found in a given text corpus.
    """

    PAD = "<PAD>"

    def __init__(self, token_to_index: Dict[str, int], index_to_token: Dict[int, str]) -> None:
        self.token_to_index = token_to_index
        self.index_to_token = index_to_token

    @staticmethod
    def _download(url: str, dataset_name: str, dest_folder: Path | str) -> Path:
        dest_folder_path = Path(dest_folder)

        dest_folder_path.mkdir(parents=True, exist_ok=True)

        filepath = dest_folder_path / f"{dataset_name}.txt"

        response = requests.get(url, stream=True)  # TODO: add timeout
        response.raise_for_status()

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return filepath

    @classmethod
    def from_corpus(cls: Type[TextCharacterVocabulary], corpus: str) -> TextCharacterVocabulary:
        vocabulary = sorted(set(corpus))
        token_to_index = {token: idx for idx, token in enumerate(vocabulary)}
        index_to_token = {idx: token for token, idx in token_to_index.items()}
        return cls(token_to_index, index_to_token)

    @classmethod
    def from_file(cls: Type[TextCharacterVocabulary], file_path: str | Path) -> TextCharacterVocabulary:
        with open(file_path, "r") as f:
            corpus = f.read()
        return cls.from_corpus(corpus)

    @classmethod
    def from_url(
        cls: Type[TextCharacterVocabulary], url: str, dataset_name: str, dest_folder: str | Path | None = None
    ) -> TextCharacterVocabulary:
        if not dest_folder:
            response = requests.get(url)
            response.raise_for_status()
            corpus = response.text
            return cls.from_corpus(corpus)

        file_path = cls._download(url, dataset_name, dest_folder)
        return cls.from_file(file_path)


Vocabularies = Union[AdderVocabulary, TextCharacterVocabulary]

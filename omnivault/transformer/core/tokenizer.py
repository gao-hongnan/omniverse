from typing import List

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

    def tokenize(self, text: str) -> List[str]:
        return list(text)  # Tokenizes the text into a list of characters

    def encode(self, text: str) -> List[int]:
        return [self.vocabulary.token_to_index.get(char, -1) for char in text]  # -1 for unknown characters

    def decode(self, tokens: List[int]) -> str:
        return "".join([self.vocabulary.index_to_token.get(token, "") for token in tokens])

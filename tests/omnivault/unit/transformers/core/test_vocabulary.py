from typing import List

import pytest

from omnivault.transformers.config.constants import TOKENS
from omnivault.transformers.config.ground_truth import GroundTruth
from omnivault.transformers.core.vocabulary import AdderVocabulary


@pytest.mark.parametrize(
    "sequence,expected_tokens",
    list(zip(GroundTruth().sequences, GroundTruth().tokenized_sequences)),
)
def test_tokenize(adder_vocab: AdderVocabulary, sequence: str, expected_tokens: List[str]) -> None:
    """Test that the sequence is tokenized as expected.

    See `GroundTruth` for the expected tokens given a sequence.
    """
    assert adder_vocab.tokenize(sequence) == expected_tokens


@pytest.mark.parametrize(
    "sequence,expected_encoded",
    list(zip(GroundTruth().sequences, GroundTruth().encoded_sequences)),
)
def test_encode(adder_vocab: AdderVocabulary, sequence: str, expected_encoded: List[int]) -> None:
    """Test that the sequence is encoded as expected.

    See `GroundTruth` for the expected encoded sequence given a sequence.
    """
    assert adder_vocab.encode(sequence) == expected_encoded


@pytest.mark.parametrize(
    "encoded_sequence,expected_decoded",
    list(zip(GroundTruth().encoded_sequences, GroundTruth().decoded_sequences)),
)
def test_decode(adder_vocab: AdderVocabulary, encoded_sequence: List[int], expected_decoded: str) -> None:
    """Test that the sequence is decoded as expected.

    See `GroundTruth` for the expected decoded sequence given a sequence.
    """
    assert adder_vocab.decode(encoded_sequence, remove_special_tokens=True) == expected_decoded


def test_encode_batch(adder_vocab: AdderVocabulary, ground_truth: GroundTruth) -> None:
    """Test that a batch of sequences is encoded as expected."""
    encoded_sequences = adder_vocab.encode_batch(ground_truth.sequences)
    assert encoded_sequences == ground_truth.encoded_sequences


def test_decode_batch(adder_vocab: AdderVocabulary, ground_truth: GroundTruth) -> None:
    """Test that a batch of sequences is decoded as expected."""
    decoded_sequences = adder_vocab.decode_batch(ground_truth.encoded_sequences)
    assert decoded_sequences == ground_truth.decoded_sequences


def test_len_vocab(adder_vocab: AdderVocabulary) -> None:
    """Test that the vocabulary length is correct."""
    assert len(adder_vocab) == len(TOKENS) == adder_vocab.vocab_size

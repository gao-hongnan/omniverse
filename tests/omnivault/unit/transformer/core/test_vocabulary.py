from typing import List

import pytest

from omnivault.transformer.config.ground_truth import GroundTruth
from omnivault.transformer.core.vocabulary import AdderVocabulary

GROUND_TRUTH = GroundTruth()


@pytest.mark.parametrize(
    "sequence,expected_tokens",
    list(zip(GROUND_TRUTH.sequences, GROUND_TRUTH.tokenized_sequences)),
)
def test_tokenize(adder_vocab: AdderVocabulary, sequence: str, expected_tokens: List[str]) -> None:
    """Test that the sequence is tokenized as expected.

    See `GroundTruth` for the expected tokens given a sequence.
    """
    assert adder_vocab.tokenize(sequence) == expected_tokens


@pytest.mark.parametrize(
    "sequence,expected_encoded",
    list(zip(GROUND_TRUTH.sequences, GROUND_TRUTH.encoded_sequences)),
)
def test_encode(adder_vocab: AdderVocabulary, sequence: str, expected_encoded: List[int]) -> None:
    """Test that the sequence is encoded as expected.

    See `GroundTruth` for the expected encoded sequence given a sequence.
    """
    assert adder_vocab.encode(sequence) == expected_encoded


@pytest.mark.parametrize(
    "encoded_sequence,expected_decoded",
    list(zip(GROUND_TRUTH.encoded_sequences, GROUND_TRUTH.decoded_sequences)),
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
    assert len(adder_vocab) == len(adder_vocab.TOKENS) == adder_vocab.vocab_size


def test_vocab_token_to_index(adder_vocab: AdderVocabulary) -> None:
    """Test that the token to index mapping is correct."""
    assert adder_vocab.token_to_index == GROUND_TRUTH.token_to_index


def test_vocab_index_to_token(adder_vocab: AdderVocabulary) -> None:
    """Test that the index to token mapping is correct."""
    assert adder_vocab.index_to_token == GROUND_TRUTH.index_to_token

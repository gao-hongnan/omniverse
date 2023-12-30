from typing import List

import pytest

from omnivault.transformer.config.ground_truth import GroundTruth
from omnivault.transformer.core.tokenizer import AdderTokenizer

GROUND_TRUTH = GroundTruth()


@pytest.mark.parametrize(
    "sequence,expected_tokens",
    list(zip(GROUND_TRUTH.sequences, GROUND_TRUTH.tokenized_sequences)),
)
def test_tokenize(adder_tokenizer: AdderTokenizer, sequence: str, expected_tokens: List[str]) -> None:
    """Test that the sequence is tokenized as expected.

    See `GroundTruth` for the expected tokens given a sequence.
    """
    assert adder_tokenizer.tokenize(sequence) == expected_tokens


@pytest.mark.parametrize(
    "sequence,expected_encoded",
    list(zip(GROUND_TRUTH.sequences, GROUND_TRUTH.encoded_sequences)),
)
def test_encode(adder_tokenizer: AdderTokenizer, sequence: str, expected_encoded: List[int]) -> None:
    """Test that the sequence is encoded as expected.

    See `GroundTruth` for the expected encoded sequence given a sequence.
    """
    assert adder_tokenizer.encode(sequence) == expected_encoded


@pytest.mark.parametrize(
    "encoded_sequence,expected_decoded",
    list(zip(GROUND_TRUTH.encoded_sequences, GROUND_TRUTH.decoded_sequences)),
)
def test_decode(adder_tokenizer: AdderTokenizer, encoded_sequence: List[int], expected_decoded: str) -> None:
    """Test that the sequence is decoded as expected.

    See `GroundTruth` for the expected decoded sequence given a sequence.
    """
    assert adder_tokenizer.decode(encoded_sequence, remove_special_tokens=True) == expected_decoded


def test_encode_batch(adder_tokenizer: AdderTokenizer, ground_truth: GroundTruth) -> None:
    """Test that a batch of sequences is encoded as expected."""
    encoded_sequences = adder_tokenizer.encode_batch(ground_truth.sequences)
    assert encoded_sequences == ground_truth.encoded_sequences


def test_decode_batch(adder_tokenizer: AdderTokenizer, ground_truth: GroundTruth) -> None:
    """Test that a batch of sequences is decoded as expected."""
    decoded_sequences = adder_tokenizer.decode_batch(ground_truth.encoded_sequences)
    assert decoded_sequences == ground_truth.decoded_sequences

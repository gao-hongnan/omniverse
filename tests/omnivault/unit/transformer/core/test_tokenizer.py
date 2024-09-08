from typing import List

import pytest

from omnivault.transformer.core.tokenizer import AdderTokenizer
from omnivault.transformer.projects.adder.snapshot import ADDER_GROUND_TRUTH, AdderGroundTruth


@pytest.mark.parametrize(
    argnames="sequence,expected_tokens",
    argvalues=list(zip(ADDER_GROUND_TRUTH.sequences, ADDER_GROUND_TRUTH.tokenized_sequences)),
)
def test_adder_tokenizer_tokenize(adder_tokenizer: AdderTokenizer, sequence: str, expected_tokens: List[str]) -> None:
    """Test that the sequence is tokenized as expected.

    See `AdderGroundTruth` for the expected tokens given a sequence.
    """
    assert adder_tokenizer.tokenize(sequence) == expected_tokens


@pytest.mark.parametrize(
    argnames="sequence,expected_encoded",
    argvalues=list(zip(ADDER_GROUND_TRUTH.sequences, ADDER_GROUND_TRUTH.encoded_sequences)),
)
def test_adder_tokenizer_encode(adder_tokenizer: AdderTokenizer, sequence: str, expected_encoded: List[int]) -> None:
    """Test that the sequence is encoded as expected.

    See `AdderGroundTruth` for the expected encoded sequence given a sequence.
    """
    assert adder_tokenizer.encode(sequence) == expected_encoded


@pytest.mark.parametrize(
    argnames="encoded_sequence,expected_decoded",
    argvalues=list(zip(ADDER_GROUND_TRUTH.encoded_sequences, ADDER_GROUND_TRUTH.decoded_sequences)),
)
def test_adder_tokenizer_decode(
    adder_tokenizer: AdderTokenizer, encoded_sequence: List[int], expected_decoded: str
) -> None:
    """Test that the sequence is decoded as expected.

    See `AdderGroundTruth` for the expected decoded sequence given a sequence.
    """
    assert adder_tokenizer.decode(encoded_sequence, remove_special_tokens=True) == expected_decoded


def test_adder_tokenizer_encode_batch(adder_tokenizer: AdderTokenizer, adder_ground_truth: AdderGroundTruth) -> None:
    """Test that a batch of sequences is encoded as expected."""
    encoded_sequences = adder_tokenizer.encode_batch(adder_ground_truth.sequences)
    assert encoded_sequences == adder_ground_truth.encoded_sequences


def test_adder_tokenizer_decode_batch(adder_tokenizer: AdderTokenizer, adder_ground_truth: AdderGroundTruth) -> None:
    """Test that a batch of sequences is decoded as expected."""
    decoded_sequences = adder_tokenizer.decode_batch(adder_ground_truth.encoded_sequences)
    assert decoded_sequences == adder_ground_truth.decoded_sequences

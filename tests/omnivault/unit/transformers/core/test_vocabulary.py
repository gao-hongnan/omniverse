from dataclasses import dataclass, field
from typing import List

import pytest
import torch

from omnivault.transformers.config.constants import TOKENS
from omnivault.transformers.core.vocabulary import AdderDataset, AdderDatasetYield, AdderVocabulary, Vocabulary


@dataclass
class GroundTruth:
    # test bad sequences like 01+02=03?
    seq_len: int = 10  # all sequences are padded to this length in this test example

    sequences: List[str] = field(default_factory=lambda: ["15+57=072", "01+02=003"])
    tokenized_sequences: List[List[str]] = field(
        default_factory=lambda: [
            ["<BOS>", "1", "5", "+", "5", "7", "=", "0", "7", "2", "<EOS>"],
            ["<BOS>", "0", "1", "+", "0", "2", "=", "0", "0", "3", "<EOS>"],
        ]
    )
    encoded_sequences: List[List[int]] = field(
        default_factory=lambda: [
            [14, 1, 5, 10, 5, 7, 13, 0, 7, 2, 15],
            [14, 0, 1, 10, 0, 2, 13, 0, 0, 3, 15],
        ]
    )
    decoded_sequences: List[str] = field(default_factory=lambda: ["15+57=072", "01+02=003"])

    inputs: List[torch.LongTensor] = field(
        default_factory=lambda: [
            torch.LongTensor([14, 1, 5, 10, 5, 7, 13, 0, 7, 2]),
            torch.LongTensor([14, 0, 1, 10, 0, 2, 13, 0, 0, 3]),
        ]
    )
    targets: List[torch.LongTensor] = field(
        default_factory=lambda: [
            torch.LongTensor([16, 16, 16, 16, 16, 16, 0, 7, 2, 15]),
            torch.LongTensor([16, 16, 16, 16, 16, 16, 0, 0, 3, 15]),
        ]
    )
    padding_masks: List[torch.BoolTensor] = field(
        default_factory=lambda: [
            torch.BoolTensor([True, True, True, True, True, True, True, True, True, True]),
            torch.BoolTensor([True, True, True, True, True, True, True, True, True, True]),
        ]
    )
    future_masks: List[torch.BoolTensor] = field(
        default_factory=lambda: [
            torch.BoolTensor(
                [
                    [True, False, False, False, False, False, False, False, False, False],
                    [True, True, False, False, False, False, False, False, False, False],
                    [True, True, True, False, False, False, False, False, False, False],
                    [True, True, True, True, False, False, False, False, False, False],
                    [True, True, True, True, True, False, False, False, False, False],
                    [True, True, True, True, True, True, False, False, False, False],
                    [True, True, True, True, True, True, True, False, False, False],
                    [True, True, True, True, True, True, True, True, False, False],
                    [True, True, True, True, True, True, True, True, True, False],
                    [True, True, True, True, True, True, True, True, True, True],
                ]
            ),
            torch.BoolTensor(
                [
                    [True, False, False, False, False, False, False, False, False, False],
                    [True, True, False, False, False, False, False, False, False, False],
                    [True, True, True, False, False, False, False, False, False, False],
                    [True, True, True, True, False, False, False, False, False, False],
                    [True, True, True, True, True, False, False, False, False, False],
                    [True, True, True, True, True, True, False, False, False, False],
                    [True, True, True, True, True, True, True, False, False, False],
                    [True, True, True, True, True, True, True, True, False, False],
                    [True, True, True, True, True, True, True, True, True, False],
                    [True, True, True, True, True, True, True, True, True, True],
                ]
            ),
        ]
    )


@pytest.fixture(scope="module")
def adder_vocab() -> Vocabulary:
    return AdderVocabulary.from_tokens(tokens=TOKENS, num_digits=2)


@pytest.fixture(scope="module")
def adder_dataset(adder_vocab: Vocabulary) -> AdderDataset[AdderDatasetYield]:
    sequences = GroundTruth().sequences
    dataset: AdderDataset[AdderDatasetYield] = AdderDataset(data=sequences, vocabulary=adder_vocab)
    return dataset


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


def test_encode_batch(adder_vocab: AdderVocabulary) -> None:
    """Test that a batch of sequences is encoded as expected."""
    encoded_sequences = adder_vocab.encode_batch(GroundTruth().sequences)
    assert encoded_sequences == GroundTruth().encoded_sequences


def test_decode_batch(adder_vocab: AdderVocabulary) -> None:
    """Test that a batch of sequences is decoded as expected."""
    decoded_sequences = adder_vocab.decode_batch(GroundTruth().encoded_sequences)
    assert decoded_sequences == GroundTruth().decoded_sequences


def test_len_vocab(adder_vocab: AdderVocabulary) -> None:
    """Test that the vocabulary length is correct."""
    assert len(adder_vocab) == len(TOKENS) == adder_vocab.vocab_size


def test_construct_future_mask(adder_dataset: AdderDataset[AdderDatasetYield]) -> None:
    """Test that the future mask is constructed correctly. Here we only test the first sequence."""
    future_mask = adder_dataset.construct_future_mask(GroundTruth().seq_len)
    torch.testing.assert_close(future_mask, GroundTruth().future_masks[0])


@pytest.mark.parametrize(
    "input,expected_padding_mask",
    list(zip(GroundTruth().inputs, GroundTruth().padding_masks)),
)
def test_construct_padding_mask(
    adder_dataset: AdderDataset[AdderDatasetYield], input: torch.LongTensor, expected_padding_mask: torch.BoolTensor
) -> None:
    """Test that the padding mask is constructed correctly. Here we only test the first sequence."""
    padding_mask = adder_dataset.construct_padding_mask(input)
    torch.testing.assert_close(padding_mask, expected_padding_mask)


@pytest.mark.parametrize(
    "encoded_sequence,expected_target",
    list(zip(GroundTruth().encoded_sequences, GroundTruth().targets)),
)
def test_construct_target(
    adder_dataset: AdderDataset[AdderDatasetYield], encoded_sequence: List[int], expected_target: torch.LongTensor
) -> None:
    """Test that the target is constructed correctly. Here we only test the first sequence."""
    target = adder_dataset.construct_target_tensor(torch.LongTensor(encoded_sequence))
    torch.testing.assert_close(target, expected_target)

@pytest.mark.parametrize(
    "encoded_sequence,expected_input",
    list(zip(GroundTruth().encoded_sequences, GroundTruth().inputs)),
)
def test_construct_input(
    adder_dataset: AdderDataset[AdderDatasetYield], encoded_sequence: torch.LongTensor, expected_input: torch.LongTensor
) -> None:
    """Test that the input is constructed correctly. Here we only test the first sequence."""
    input = adder_dataset.construct_input_tensor(torch.LongTensor(encoded_sequence))
    torch.testing.assert_close(input, expected_input)


def test_dataset_integration_with_getitem(adder_dataset: AdderDataset[AdderDatasetYield]) -> None:
    """Test that the dataset returns the correct item."""
    index = 0
    length = len(adder_dataset)
    for input, target, padding_mask, future_mask in adder_dataset:  # type: ignore[attr-defined]
        torch.testing.assert_close(input, GroundTruth().inputs[index])
        torch.testing.assert_close(target, GroundTruth().targets[index])
        torch.testing.assert_close(padding_mask, GroundTruth().padding_masks[index])
        torch.testing.assert_close(future_mask, GroundTruth().future_masks[index])

        assert input.shape == (10,)
        assert target.shape == (10,)
        assert padding_mask.shape == (10,)
        assert future_mask.shape == (10, 10)

        assert input.dtype == torch.long
        assert target.dtype == torch.long
        assert padding_mask.dtype == torch.bool
        assert future_mask.dtype == torch.bool

        assert isinstance(input, torch.LongTensor)
        assert isinstance(target, torch.LongTensor)
        assert isinstance(padding_mask, torch.BoolTensor)
        assert isinstance(future_mask, torch.BoolTensor)

        index += 1
        if index == length:
            break

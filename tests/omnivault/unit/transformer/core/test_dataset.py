from typing import List

import pytest
import torch
from torch.utils.data import Subset

from omnivault.transformer.config.ground_truth import GroundTruth
from omnivault.transformer.core.dataset import AdderDataset, AdderDatasetYield, collate_fn, split_dataset

GROUND_TRUTH = GroundTruth()


def test_construct_future_mask(adder_dataset: AdderDataset, ground_truth: GroundTruth) -> None:
    """Test that the future mask is constructed correctly. Here we only test the first sequence."""
    future_mask = adder_dataset.construct_future_mask(ground_truth.seq_len)
    torch.testing.assert_close(future_mask, ground_truth.future_masks[0])


@pytest.mark.parametrize(
    "input,expected_padding_mask",
    list(zip(GROUND_TRUTH.inputs, GROUND_TRUTH.padding_masks)),
)
def test_construct_padding_mask(
    adder_dataset: AdderDataset, input: torch.LongTensor, expected_padding_mask: torch.BoolTensor
) -> None:
    """Test that the padding mask is constructed correctly. Here we only test the first sequence."""
    padding_mask = adder_dataset.construct_padding_mask(input)
    torch.testing.assert_close(padding_mask, expected_padding_mask)


@pytest.mark.parametrize(
    "encoded_sequence,expected_target",
    list(zip(GROUND_TRUTH.encoded_sequences, GROUND_TRUTH.targets)),
)
def test_construct_target(
    adder_dataset: AdderDataset, encoded_sequence: List[int], expected_target: torch.LongTensor
) -> None:
    """Test that the target is constructed correctly. Here we only test the first sequence."""
    target = adder_dataset.construct_target_tensor(torch.LongTensor(encoded_sequence))
    torch.testing.assert_close(target, expected_target)


@pytest.mark.parametrize(
    "encoded_sequence,expected_input",
    list(zip(GROUND_TRUTH.encoded_sequences, GROUND_TRUTH.inputs)),
)
def test_construct_input(
    adder_dataset: AdderDataset, encoded_sequence: torch.LongTensor, expected_input: torch.LongTensor
) -> None:
    """Test that the input is constructed correctly. Here we only test the first sequence."""
    input = adder_dataset.construct_input_tensor(torch.LongTensor(encoded_sequence))
    torch.testing.assert_close(input, expected_input)


def test_dataset_integration_with_getitem(adder_dataset: AdderDataset, ground_truth: GroundTruth) -> None:
    """Test that the dataset returns the correct item."""
    index = 0
    length = len(adder_dataset)
    for input, target, padding_mask, future_mask in adder_dataset:  # type: ignore[attr-defined]
        torch.testing.assert_close(input, ground_truth.inputs[index])
        torch.testing.assert_close(target, ground_truth.targets[index])
        torch.testing.assert_close(padding_mask, ground_truth.padding_masks[index])
        torch.testing.assert_close(future_mask, ground_truth.future_masks[index])

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


def test_collate_fn(mock_batch: List[AdderDatasetYield], ground_truth: GroundTruth) -> None:
    # Call your collate_fn function with the mock_batch

    inputs_padded, targets_padded, padding_masks_padded_and_expanded, future_masks_expanded = collate_fn(
        mock_batch, **ground_truth.collate_fn
    )

    # Assert that the first dimension of the output tensors equals the batch size
    assert inputs_padded.shape[0] == len(mock_batch)
    assert targets_padded.shape[0] == len(mock_batch)
    assert padding_masks_padded_and_expanded.shape[0] == len(mock_batch)
    assert future_masks_expanded.shape[0] == len(mock_batch)

    torch.testing.assert_close(inputs_padded, ground_truth.inputs_collated)
    torch.testing.assert_close(targets_padded, ground_truth.targets_collated)
    torch.testing.assert_close(padding_masks_padded_and_expanded, ground_truth.padding_masks_collated)
    torch.testing.assert_close(future_masks_expanded, ground_truth.future_masks_collated)

    # Check if the padding has been applied correctly
    for i, (input, target, _, _) in enumerate(mock_batch):
        # Check for padding tokens in inputs and targets
        input_len, target_len = input.size(0), target.size(0)
        assert torch.all(inputs_padded[i, :input_len] == input)
        assert torch.all(targets_padded[i, :target_len] == target)

        # Check padding mask shape and values
        assert padding_masks_padded_and_expanded[i, :, :, :input_len].all()
        assert not padding_masks_padded_and_expanded[i, :, :, input_len:].any()

        # Check future mask shape
        assert future_masks_expanded[i].size() == (1, input_len, input_len)


@pytest.mark.parametrize("split", [[0.7, 0.1, 0.2], [0.8, 0.1, 0.1], [0.6, 0.2, 0.2]])
@pytest.mark.parametrize("seed", [42, 1992])
def test_split_dataset(adder_dataset_but_larger: AdderDataset, split: List[float], seed: int) -> None:
    """Test splitting the dataset into train, validation, and test sets."""
    # Perform the split
    train_dataset, val_dataset, test_dataset = split_dataset(adder_dataset_but_larger, split, seed)

    # Assert that the datasets are of type Subset
    assert isinstance(train_dataset, Subset)
    assert isinstance(val_dataset, Subset)
    assert isinstance(test_dataset, Subset)

    # Calculate expected lengths
    total_len = len(adder_dataset_but_larger)
    expected_train_len = int(total_len * split[0])
    expected_val_len = int(total_len * split[1])
    expected_test_len = total_len - expected_train_len - expected_val_len

    # Assert that the datasets are split correctly
    assert len(train_dataset) == expected_train_len
    assert len(val_dataset) == expected_val_len
    assert len(test_dataset) == expected_test_len

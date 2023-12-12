from typing import List

import pytest
import torch

from omnivault.transformer.core.dataset import AdderDataset, AdderDatasetYield

from omnivault.transformer.config.ground_truth import GroundTruth




def test_construct_future_mask(adder_dataset: AdderDataset[AdderDatasetYield], ground_truth: GroundTruth) -> None:
    """Test that the future mask is constructed correctly. Here we only test the first sequence."""
    future_mask = adder_dataset.construct_future_mask(ground_truth.seq_len)
    torch.testing.assert_close(future_mask, ground_truth.future_masks[0])


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


def test_dataset_integration_with_getitem(adder_dataset: AdderDataset[AdderDatasetYield], ground_truth: GroundTruth) -> None:
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


# import pytest
# import torch
# from torch.utils.data.dataloader import default_collate

# # Assuming AdderVocabulary and other necessary imports and fixtures

# @pytest.fixture
# def mock_batch():
#     # Create a mock batch of data
#     # Ensure that this mock data aligns with what your real data would look like
#     return [
#         (torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), torch.tensor([True, False, True]), torch.tensor([True, True, False])),
#         (torch.tensor([1, 2]), torch.tensor([4, 5]), torch.tensor([True, False]), torch.tensor([True, False])),
#         # Add more tuples to represent additional data points
#     ]

# def test_collate_fn(mock_batch):
#     # Call your collate_fn function with the mock_batch
#     batch_first = True
#     pad_token_id = 16  # Use the appropriate pad token id for your dataset
#     inputs_padded, targets_padded, padding_masks_padded, future_masks_expanded = collate_fn(mock_batch, batch_first, pad_token_id)

#     # Assert the shapes of the returned tensors
#     assert inputs_padded.shape[0] == len(mock_batch)
#     # Add more assertions for shape, type, and values as needed

#     # Optionally, you can compare the output of your custom collate_fn with the default_collate to see if they behave similarly
#     default_collated = default_collate(mock_batch)
#     assert torch.equal(inputs_padded, default_collated[0])
#     # Add more comparisons as needed

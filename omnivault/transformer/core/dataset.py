from __future__ import annotations

from typing import Any, Dict, List, Tuple, TypeVar, cast

import torch
from rich.pretty import pprint
from torch.utils.data import DataLoader, Dataset, Subset

from omnivault._types._alias import NotGiven
from omnivault._types._sentinel import NOT_GIVEN
from omnivault.transformer.config.composer import Composer
from omnivault.transformer.core.vocabulary import AdderVocabulary, Vocabulary

# FIXME: Should we rename `AdderDatasetYield` to `DatasetYield` to be more generic?
AdderDatasetYield = Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor, torch.BoolTensor]
AdderDataset_co = TypeVar("AdderDataset_co", bound=AdderDatasetYield, covariant=True)


# TODO: ideally splitting data should be done within the dataset class to
# speed up the process, as we only need to load the data that we need in memory.
# See Kapathy's https://github.com/karpathy/minGPT/tree/master/projects/adder.
class AdderDataset(Dataset[AdderDataset_co]):
    """
    A Dataset class for encoding sequences for an addition problem.

    This dataset class takes a list of string representations of addition
    problems and a vocabulary object. It encodes the problems and prepares
    input and target tensors for model training.

    Parameters
    ----------
    data : List[str]
        A list of strings, each representing an addition problem, e.g., "15+57=072".
    vocabulary : Vocabulary
        A `Vocabulary` object used for encoding the strings into numerical tokens.
    """

    def __init__(self, data: List[str], vocabulary: Vocabulary) -> None:
        super().__init__()

        self.data = data
        self.vocabulary = vocabulary

        self.equal_token_id: int = vocabulary.token_to_index[AdderVocabulary.EQUAL]
        self.pad_token_id: int = vocabulary.token_to_index[AdderVocabulary.PAD]

    def __len__(self) -> int:
        return len(self.data)

    def construct_future_mask(self, seq_len: int) -> torch.BoolTensor:
        future_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1).to(torch.bool)
        future_mask = future_mask.contiguous()
        future_mask = future_mask == 0
        return torch.BoolTensor(future_mask)

    def construct_padding_mask(self, input_sequence: torch.Tensor) -> torch.BoolTensor:
        padding_mask = input_sequence != self.pad_token_id
        return torch.BoolTensor(padding_mask)

    def construct_input_tensor(self, input_sequence: torch.Tensor) -> torch.LongTensor:
        # Returns all but the last token
        return torch.LongTensor(input_sequence[:-1])

    def construct_target_tensor(self, input_sequence: torch.Tensor) -> torch.LongTensor:
        # Masks out tokens before the equal sign
        # TODO: is clone replaceable? it is not removable for now due to mutation.
        target = input_sequence.clone()
        where_equal_index = torch.where(input_sequence == self.equal_token_id)[0].item()
        where_equal_index = int(where_equal_index)  # to appease mypy lol
        target[: where_equal_index + 1] = self.pad_token_id
        return torch.LongTensor(target[1:])

    def __getitem__(self, index: int) -> AdderDataset_co:
        """
        data = ["15+57=072", "92+00=092", "95+53=148", "15+10=025"]
        getitem selects one index randomly, say 2, to obtain
        data[2] -> "95+53=148"
        we need split to input and target by slicing
        x takes all but last token
        y takes all but first token
        """
        raw_sequence: str = self.data[index]
        encoded_sequence: List[int] = self.vocabulary.encode(raw_sequence)

        input_sequence: torch.LongTensor = torch.tensor(encoded_sequence, dtype=torch.long)  # type: ignore[assignment]

        input = self.construct_input_tensor(input_sequence)  # x
        target = self.construct_target_tensor(input_sequence)  # y
        padding_mask = self.construct_padding_mask(input)
        future_mask = self.construct_future_mask(input.size(0))
        return cast(AdderDataset_co, (input, target, padding_mask, future_mask))  # TODO: really mypy?


def collate_fn(
    batch: List[AdderDatasetYield],
    batch_first: bool = True,
    pad_token_id: int = 0,
) -> AdderDatasetYield:
    """
    Collates a batch of data into padded tensors for input, target, and masks.

    This function takes a batch of data produced by `AdderDataset` instances and
    pads the sequences to the same length, preparing them for batched processing
    in models. It also reshapes padding and future masks to match the batch
    dimensions.

    Parameters
    ----------
    batch : List[AdderDatasetYield]
        A batch of data, where each item is a tuple containing input, target,
        padding mask, and future mask tensors.
    batch_first : bool, default=True
        If `batch_first=True`, the resulting tensor
        `inputs_padded` will have a shape of `(batch_size, max_seq_len)`, where
        `batch_size` is the number of samples in the batch and `max_seq_len` is the
        length of the longest sequence in the batch. If `batch_first=False`, the shape
        will be `(max_seq_len, batch_size)`.
    pad_token_id : int, default=0
        The `padding_value` parameter specifies the value
        to use for padding shorter sequences. `pad_token_id` is typically used here,
        which should correspond to the padding token's ID in your vocabulary.

    Returns
    -------
    AdderDatasetYield
        A tuple containing the following tensors:
        - Padded input tensor
        - Padded target tensor
        - Reshaped padding mask tensor
        - Reshaped and expanded future mask tensor

    Notes
    -----
    The function assumes that all sequences in the batch are of the same type
    and can be padded with the same `pad_token_id`.
    """
    # omega confused during zipping so put here for clarity
    # that when you unzip a batch it becomes a tuple.
    # inputs: Tuple[torch.Tensor, ...]
    # targets: Tuple[torch.Tensor, ...]
    # padding_masks: Tuple[torch.Tensor, ...]
    # future_masks: Tuple[torch.Tensor, ...]

    inputs, targets, padding_masks, future_masks = zip(*batch)

    # Padding sequences to the same length: convert to list to appease typing
    # of pad_sequence as it expects a list of tensors or tensors, not tuple.
    inputs_padded: torch.Tensor = torch.nn.utils.rnn.pad_sequence(
        list(inputs), batch_first=batch_first, padding_value=pad_token_id
    )
    targets_padded: torch.Tensor = torch.nn.utils.rnn.pad_sequence(
        list(targets), batch_first=batch_first, padding_value=pad_token_id
    )
    padding_masks_padded: torch.Tensor = torch.nn.utils.rnn.pad_sequence(
        list(padding_masks), batch_first=batch_first, padding_value=pad_token_id
    )

    # Reshaping padding masks
    batch_size, seq_len = inputs_padded.size(0), inputs_padded.size(1)

    # padding_masks before view has shape: (batch_size, seq_len)
    # we want it to be (B, L, L) then (B, 1, L, L)
    padding_masks_padded_and_expanded = padding_masks_padded.view(batch_size, 1, 1, seq_len).expand(
        batch_size, 1, seq_len, seq_len
    )

    # future mask has shape (L, L) but we want it to be (B, L, L) then (B, 1, L, L)
    future_masks: torch.BoolTensor = torch.stack(future_masks)  # type: ignore[assignment,no-redef]

    future_masks_expanded = future_masks.expand(batch_size, -1, -1).unsqueeze(1)  # type: ignore[attr-defined]
    return cast(
        AdderDatasetYield, (inputs_padded, targets_padded, padding_masks_padded_and_expanded, future_masks_expanded)
    )


def create_loader(
    dataset: AdderDataset[AdderDatasetYield],
    loader_config: Dict[str, Any],
    collate_fn_config: Dict[str, Any] | NotGiven = NOT_GIVEN,
) -> DataLoader[AdderDatasetYield]:
    if isinstance(
        collate_fn_config, NotGiven
    ):  # TODO: excuse me mypy, why cannot I do if collate_fn_config is NOT_GIVEN?
        collate_fn_config = {"batch_first": True, "pad_token_id": 0}
    return DataLoader(
        dataset=dataset,
        collate_fn=lambda batch: collate_fn(batch, **collate_fn_config),  # type: ignore[arg-type]
        **loader_config,
    )


def split_dataset(
    dataset: AdderDataset[AdderDatasetYield], split: List[float], seed: int
) -> Tuple[
    Subset[AdderDatasetYield], Subset[AdderDatasetYield], Subset[AdderDatasetYield]
]:  # TODO: unclean since it should return AdderDataset[AdderDatasetYield] but mypy is not happy
    # if sum(split) != 1.0:
    #     raise ValueError(f"Split ratios should sum to 1 but got {sum(split)}.")

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        lengths=split,
        generator=torch.Generator().manual_seed(seed),
    )
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    config = Composer()
    vocab = AdderVocabulary.from_tokens(tokens=config.constants.TOKENS, num_digits=config.constants.NUM_DIGITS)

    pprint(vocab.token_to_index)
    pprint(vocab.index_to_token)

    pprint(vocab.encode("1"))
    pprint(vocab.encode("+"))

    sequence = "15+57=072"
    sequences = ["15+57=072", "01+02=003"]  # , "95+53=148", "15+10=025"]

    encoded_sentence = vocab.encode(sequence)
    print(f"Encoded sentence: {encoded_sentence}")
    decoded_sentence = vocab.decode(encoded_sentence)
    pprint(decoded_sentence)

    encoded_sentences = vocab.encode_batch(sequences)  # type: ignore[attr-defined]
    pprint(encoded_sentences)
    decoded_sentences = vocab.decode_batch(encoded_sentences)  # type: ignore[attr-defined]
    pprint(decoded_sentences)

    dataset: AdderDataset[AdderDatasetYield] = AdderDataset(data=sequences, vocabulary=vocab)

    print()

    counter = 1
    for x, y, pad_mask, future_mask in dataset:  # type: ignore[attr-defined]
        print("x")
        pprint(x)
        pprint(isinstance(x, torch.LongTensor))

        print("y")
        pprint(y)
        print("pad")
        pprint(pad_mask)
        print("future")
        pprint(future_mask)
        if counter == 2:
            break
    # at this junction it is possible for the seq len
    # to vary. Dataset only cares about generating 1 single
    # sample data point and do not worry about different
    # sequence length across other samples.
    # but in torch we train via batches, and with different
    # batch sizes we may encounter issues like you know
    # matrix multiplication may not work.

    # As we see later, the collate fn will be passed into
    # dataloader. where dataloader gather individual samples
    # from dataset into BATCHES \mathcal{B}. But they
    # dont care if your individual samples from dataset
    # is of diff length, or if you want to broadcast some
    # padding or future mask TO BE THE SAME AS BATCH SIZE
    # IN SOME DIMENSION.

    # The `collate_fn` defines how to combine these variable-length samples into a
    # batch. This usually involves padding the sequences in the batch to a common
    # length, which is typically the length of the longest sequence in the batch.

    # Assuming your dataset is initialized as `my_dataset`
    # and your `collate_fn` is defined as shown above
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=lambda batch: collate_fn(batch, batch_first=True, pad_token_id=16),
    )

    for i, batch in enumerate(dataloader):
        (
            inputs_padded,
            targets_padded,
            padding_masks_padded_and_expanded,
            future_masks_expanded,
        ) = batch

        # Print shapes
        print(f"Batch {i+1}")
        print("Inputs Shape:", inputs_padded.shape)
        print("Targets Shape:", targets_padded.shape)
        print("Padding Masks Shape:", padding_masks_padded_and_expanded.shape)
        print("Future Masks Shape:", future_masks_expanded.shape)

        # Print values (consider printing only a part of each tensor for large datasets)
        print("Inputs Values:", inputs_padded)
        print("Targets Values:", targets_padded)
        print("Padding Masks Values:", padding_masks_padded_and_expanded)
        print("Future Masks Values:", future_masks_expanded)

        # Add a separator for readability between batches
        print("-" * 50)

        # Optionally, break after a few batches to avoid too much output
        if i >= 2:  # Change this number based on how many batches you want to inspect
            break

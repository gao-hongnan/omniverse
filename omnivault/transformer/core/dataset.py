from __future__ import annotations

from typing import Any, Dict, List, Literal, Tuple, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from omnivault._types._alias import NotGiven
from omnivault._types._sentinel import NOT_GIVEN
from omnivault.transformer.core.tokenizer import AdderTokenizer, TextCharacterTokenizer
from omnivault.transformer.core.vocabulary import AdderVocabulary

# TODO: if both yield are same, then no point having the union of them except for semantics.
#       to fix this simply only use DatasetYield = Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor, torch.BoolTensor]
#       and remove the union and the typevar.
AdderDatasetYield = Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor, torch.BoolTensor]
TextCharacterDatasetYield = Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor, torch.BoolTensor]
DatasetYield = Union[AdderDatasetYield, TextCharacterDatasetYield]
Dataset_co = TypeVar(
    "Dataset_co", bound=DatasetYield, covariant=True
)  # using covariant as pytorch Dataset is covariant in its type parameter


def get_batch(
    *,
    dataset: npt.NDArray[np.uint16],
    batch_size: int,
    context_length: int,
    generator: torch.Generator | None = None,
    device_type: Literal["cpu", "cuda"] = "cpu",  # exclude mps
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a batch of input-output pairs from the given dataset.

    Parameters
    ----------
    dataset : numpy.ndarray
        The dataset to generate batches from. It should be a 1D NumPy array of type uint16.
    batch_size : int
        The number of samples in each batch.
    context_length : int
        The length of the context window for each sample.
    generator : torch.Generator, optional
        The PyTorch generator to use for generating random indices. If not provided,
        no seed is set.
    device_type : str, optional
        The device type to use for the generated tensors. Can be either "cpu" or "cuda".
        Defaults to "cpu".

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple containing two tensors:
        - x: The input tensor of shape (batch_size, context_length).
        - y: The output tensor of shape (batch_size, context_length), where each element
             is shifted by 1 compared to the corresponding element in x.

    Notes
    -----
    - The function assumes that the dataset is a memory-mapped array to avoid memory leaks.
    - If the device type is "cuda", the function uses pinned memory for faster data transfer
      to the GPU.

    Examples
    --------
    >>> dataset = np.arange(100, dtype=np.uint16)
    >>> batch_size = 4
    >>> context_length = 10
    >>> x, y = get_batch(dataset=dataset, batch_size=batch_size, context_length=context_length)
    >>> x.shape
    torch.Size([4, 10])
    >>> y.shape
    torch.Size([4, 10])
    """
    # Source: Karpathy, We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122

    # if not isinstance(dataset, np.memmap):
    #     raise ValueError("The dataset should be a memory-mapped array. Example: data = np.memmap('data.npy', dtype=np.uint16, mode='r')")

    device = torch.device("cuda") if device_type == "cuda" else torch.device("cpu")
    low, high = 0, len(dataset) - context_length
    size = (batch_size,)
    indices = torch.randint(low=low, high=high, size=size, generator=generator)

    x = torch.stack([torch.from_numpy((dataset[index : index + context_length]).astype(np.int64)) for index in indices])
    y = torch.stack(
        [torch.from_numpy((dataset[index + 1 : index + 1 + context_length]).astype(np.int64)) for index in indices]
    )
    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


class BaseDataset(Dataset[Dataset_co]):
    """Dataset base class, currently not filled in, but act as a conveyor for
    us to use Dataset[Dataset_co] as type hinting."""


# TODO: ideally splitting data should be done within the dataset class to
# speed up the process, as we only need to load the data that we need in memory.
# See Kapathy's https://github.com/karpathy/minGPT/tree/master/projects/adder.
class AdderDataset(BaseDataset[AdderDatasetYield]):
    """
    A Dataset class for encoding sequences for an addition problem.

    This dataset class takes a list of string representations of addition
    problems and a vocabulary object. It encodes the problems and prepares
    input and target tensors for model training.

    Parameters
    ----------
    data : List[str]
        A list of strings, each representing an addition problem, e.g., "15+57=072".
    tokenizer : AdderTokenizer
        A `AdderTokenizer` object used for encoding the strings into numerical tokens
        and decoding the numerical tokens back into strings.

    Attributes
    ----------
    vocabulary : Vocabulary
        A `Vocabulary` object used for encoding the strings into numerical tokens.
    equal_token_id : int
        The numerical token ID for the equal sign.
    pad_token_id : int
        The numerical token ID for the padding token.
    """

    def __init__(self, data: List[str], tokenizer: AdderTokenizer) -> None:
        super().__init__()

        self.data = data
        self.tokenizer = tokenizer
        self.vocabulary = tokenizer.vocabulary

        self.equal_token_id: int = self.vocabulary.token_to_index[AdderVocabulary.EQUAL]
        self.pad_token_id: int = self.vocabulary.token_to_index[AdderVocabulary.PAD]

    def __len__(self) -> int:
        return len(self.data)

    def construct_future_mask(self, seq_len: int) -> torch.BoolTensor:
        future_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=0).to(torch.bool)
        future_mask = future_mask.contiguous()
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

    def __getitem__(self, index: int) -> AdderDatasetYield:
        """
        data = ["15+57=072", "92+00=092", "95+53=148", "15+10=025"]
        getitem selects one index randomly, say 2, to obtain
        data[2] -> "95+53=148"
        we need split to input and target by slicing
        x takes all but last token
        y takes all but first token
        """
        raw_sequence: str = self.data[index]
        encoded_sequence: List[int] = self.tokenizer.encode(raw_sequence)

        input_sequence: torch.LongTensor = torch.tensor(encoded_sequence, dtype=torch.long)  # type: ignore[assignment]

        input = self.construct_input_tensor(input_sequence)  # x
        target = self.construct_target_tensor(input_sequence)  # y
        padding_mask = self.construct_padding_mask(input)
        future_mask = self.construct_future_mask(input.size(0))
        return cast(AdderDatasetYield, (input, target, padding_mask, future_mask))  # TODO: really mypy?


class TextCharacterDataset(BaseDataset[TextCharacterDatasetYield]):
    """
    A Dataset class for encoding sequences from a text corpus at the character
    level.

    This dataset class takes a text corpus and a vocabulary object. It encodes the
    text corpus into numerical tokens and prepares input and target tensors for
    model training, based on the specified context length.

    Parameters
    ----------
    text_corpus : str
        A string representing the entire text corpus.
    vocabulary : List[str]
        A list of unique characters representing the vocabulary.
    context_length : int
        The length of the context window used for creating each training example.
    """

    def __init__(self, corpus: str, context_length: int, tokenizer: TextCharacterTokenizer) -> None:
        super().__init__()

        self.corpus = corpus
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.vocabulary = tokenizer.vocabulary

    def construct_future_mask(self, seq_len: int) -> torch.BoolTensor:
        future_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=0).to(torch.bool)
        future_mask = future_mask.contiguous()
        return torch.BoolTensor(future_mask)

    @property
    def corpus_size(self) -> int:
        return len(self.corpus)

    def __len__(self) -> int:
        return len(self.corpus) - self.context_length

    def __getitem__(self, index: int) -> TextCharacterDatasetYield:
        """
        Retrieves a training example based on the specified index.

        The method selects a context window from the text corpus and encodes it into numerical tokens.
        It then splits the encoded context into input and target tensors for training.

        Parameters
        ----------
        index : int
            The index at which to start the context window.

        Returns
        -------
        Tuple[torch.LongTensor, torch.LongTensor]
            A tuple containing the input and target tensors.
        """
        context = self.corpus[index : index + self.context_length + 1]
        context_encoded = self.tokenizer.encode(context)
        x = torch.tensor(context_encoded[:-1], dtype=torch.long)
        y = torch.tensor(context_encoded[1:], dtype=torch.long)

        padding_mask = torch.ones_like(x, dtype=torch.bool)

        seq_len = x.size(0)
        assert seq_len == self.context_length, f"seq_len {seq_len} != context_length {self.context_length}"
        future_mask = self.construct_future_mask(seq_len)

        return cast(TextCharacterDatasetYield, (x, y, padding_mask, future_mask))


def construct_dummy_batch_future_masks(batch_size: int, seq_len: int) -> torch.BoolTensor:
    """Broadcast future mask from shape (L, L) to (B, L, L) then (B, 1, L, L)."""
    # Create a lower triangular mask for a single sequence
    future_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=0).to(torch.bool)
    future_mask = future_mask.contiguous()
    # broadcast future mask from shape (L, L) to (B, L, L)
    future_masks = future_mask.unsqueeze(0).expand(batch_size, -1, -1)
    # broadcast future mask from shape (B, L, L) to (B, 1, L, L)
    future_masks = future_masks.unsqueeze(1)
    return torch.BoolTensor(future_masks)


def construct_dummy_batch_target_padding_masks(batch_size: int, seq_len: int) -> torch.BoolTensor:
    """Construct a dummy batch of target padding masks of shape (B, 1, L, L) which
    assumes there is no padding token involved."""

    return torch.BoolTensor(torch.ones((batch_size, 1, seq_len, seq_len), dtype=torch.bool))


def collate_fn(
    batch: List[DatasetYield],
    batch_first: bool = True,
    pad_token_id: int = 0,
) -> DatasetYield:
    """
    Collates a batch of data into padded tensors for input, target, and masks.

    This function takes a batch of data produced by `AdderDataset` instances and
    pads the sequences to the same length, preparing them for batched processing
    in models. It also reshapes padding and future masks to match the batch
    dimensions.

    Parameters
    ----------
    batch : List[DatasetYield]
        A batch of data, where each item is a tuple containing input, target,
        padding mask, and future mask tensors.
    batch_first : bool, default=True
        If `batch_first=True`, the resulting tensor `inputs_padded` will have a shape of
        `(batch_size, max_seq_len)`, where `batch_size` is the number of samples in the
        batch and `max_seq_len` is the length of the longest sequence in the batch. If
        `batch_first=False`, the shape will be `(max_seq_len, batch_size)`. Note here
        `max_seq_len` is similar to `context_length` in our terminology.
    pad_token_id : int, default=0
        The `padding_value` parameter specifies the value to use for padding shorter
        sequences. `pad_token_id` is typically used here, which should correspond to the
        padding token's ID in your vocabulary.

    Returns
    -------
    DatasetYield
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
    dataset: Dataset[Dataset_co],
    loader_config: Dict[str, Any],
    collate_fn_config: Dict[str, Any] | NotGiven = NOT_GIVEN,
) -> DataLoader[Dataset_co]:
    if collate_fn_config is NOT_GIVEN:
        return DataLoader(
            dataset=dataset,
            **loader_config,
        )
    return DataLoader(
        dataset=dataset,
        collate_fn=lambda batch: collate_fn(batch, **collate_fn_config),  # type: ignore[arg-type]
        **loader_config,
    )


def split_dataset(
    dataset: Dataset[Dataset_co], split: List[float], seed: int
) -> Tuple[
    Subset[DatasetYield], Subset[DatasetYield], Subset[DatasetYield]
]:  # TODO: unclean since it should return AdderDataset[AdderDatasetYield] but mypy is not happy
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        lengths=split,
        generator=torch.Generator().manual_seed(seed),
    )
    return train_dataset, val_dataset, test_dataset

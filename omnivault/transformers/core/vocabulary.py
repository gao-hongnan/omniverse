from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Type

import torch
from rich.pretty import pprint
from torch.utils.data import DataLoader, Dataset

from omnivault.transformers.config.constants import TOKENS


class Vocabulary(ABC):
    # Special tokens as class attributes
    BOS = "<BOS>"
    EOS = "<EOS>"
    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(
        self,
        token_to_index: Dict[str, int],
        index_to_token: Dict[int, str],
        num_digits: int,
    ) -> None:
        self.token_to_index = token_to_index
        self.index_to_token = index_to_token
        self.num_digits = num_digits

    @abstractmethod
    def tokenize(self, sequence: str, add_special_tokens: bool = True) -> List[str]:
        """
        Tokenizes a sequence into a sequence (list) of tokens.

        Example
        -------
        >>> tokenizer = Tokenizer()
        >>> tokenizer.tokenize("The quick brown fox jumps over the lazy dog.")
        ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "."]

        Parameters
        ----------
        sequence : str
            The sequence to tokenize.
        add_special_tokens : bool, optional
            Whether to add special tokens to the sequence of tokens, by default True.

        Returns
        -------
        List[str]
            The sequence of tokens.
        """

    @abstractmethod
    def encode(self, sequence: str) -> int:
        """
        Encodes a sequence to its corresponding integer index.

        Parameters
        ----------
        sequence : str
            The sequence to encode.

        Returns
        -------
        int
            The integer index corresponding to the token.
        """

    @abstractmethod
    def decode(self, sequence: str, remove_special_tokens: bool = True) -> str:
        """
        Decodes an integer index back to its corresponding token.

        Parameters
        ----------
        index : int
            The integer index to decode.

        Returns
        -------
        str
            The token corresponding to the integer index.
        """

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of tokens in the vocabulary.

        Returns
        -------
        int
            The number of tokens in the vocabulary.
        """

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """
        Returns the size of the vocabulary.

        Returns
        -------
        int
            The size of the vocabulary.
        """

    @classmethod
    def from_tokens(
        cls: Type[Vocabulary], tokens: List[str], num_digits: int = 2
    ) -> Vocabulary:
        token_to_index = {token: idx for idx, token in enumerate(tokens)}
        index_to_token = {idx: token for token, idx in token_to_index.items()}
        return cls(token_to_index, index_to_token, num_digits)


class AdderVocabulary(Vocabulary):
    # special tokens
    BOS = "<BOS>"
    EOS = "<EOS>"
    PAD = "<PAD>"
    UNK = "<UNK>"
    EQUAL = "="
    TOKENS = TOKENS

    def tokenize(self, sequence: str, add_special_tokens: bool = True) -> List[str]:
        tokens = [char for char in sequence]
        if add_special_tokens:
            tokens = [AdderVocabulary.BOS] + tokens + [AdderVocabulary.EOS]
        return tokens

    def encode(self, sequence: str, add_special_tokens: bool = True) -> List[int]:
        tokens: List[str] = self.tokenize(
            sequence, add_special_tokens=add_special_tokens
        )
        return [
            self.token_to_index.get(token, self.token_to_index[AdderVocabulary.UNK])
            for token in tokens
        ]

    def encode_batch(
        self, sequences: List[str], add_special_tokens: bool = True
    ) -> List[List[int]]:
        return [
            self.encode(sequence, add_special_tokens=add_special_tokens)
            for sequence in sequences
        ]

    def decode(self, sequence: str, remove_special_tokens: bool = True) -> str:
        decoded = "".join(
            [self.index_to_token.get(char, AdderVocabulary.UNK) for char in sequence]
        )

        if remove_special_tokens:
            decoded = re.sub(
                f"{AdderVocabulary.BOS}|{AdderVocabulary.EOS}|{AdderVocabulary.PAD}|{AdderVocabulary.UNK}",
                "",
                decoded,
            )
        return decoded

    def decode_batch(
        self, sequences: List[List[int]], remove_special_tokens: bool = True
    ) -> List[str]:
        return [
            self.decode(sequence, remove_special_tokens=remove_special_tokens)
            for sequence in sequences
        ]

    def __len__(self) -> int:
        return len(self.token_to_index)

    @property
    def vocab_size(self) -> int:
        return len(self)


class AdderDataset(Dataset):
    def __init__(self, data: List[str], vocabulary: Vocabulary) -> None:
        super().__init__()

        self.data = data
        self.vocabulary = vocabulary

        self.equal_token_id: int = vocabulary.token_to_index[AdderVocabulary.EQUAL]
        self.pad_token_id: int = vocabulary.token_to_index[AdderVocabulary.PAD]

    def __len__(self) -> int:
        return len(self.data)

    def construct_future_mask(self, seq_len: int) -> torch.BoolTensor:
        future_mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1
        ).to(torch.bool)
        future_mask = future_mask.contiguous()
        return future_mask == 0

    def construct_padding_mask(self, input_sequence: torch.Tensor) -> torch.BoolTensor:
        padding_mask = input_sequence != self.pad_token_id
        return padding_mask

    def construct_input_tensor(self, input_sequence: torch.Tensor) -> torch.LongTensor:
        # Returns all but the last token
        return input_sequence[:-1]

    def construct_target_tensor(self, input_sequence: torch.Tensor) -> torch.LongTensor:
        # Masks out tokens before the equal sign
        # TODO: is clone replaceable? it is not removable for now due to mutation.
        target = input_sequence.clone()
        where_equal_index = torch.where(input_sequence == self.equal_token_id)[0].item()
        target[: where_equal_index + 1] = self.pad_token_id
        return target[1:]

    def __getitem__(self, index: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        data = ["15+57=072", "92+00=092", "95+53=148", "15+10=025"]
        getitem selects one index randomly, say 2, to obtain
        data[2] -> "95+53=148"
        we need split to input and target by slicing
        x takes all but last token
        y takes all but first token
        """
        input_sequence: str = self.data[index]
        input_sequence: List[int] = self.vocabulary.encode(input_sequence)

        input_sequence: torch.LongTensor = torch.tensor(
            input_sequence, dtype=torch.long
        )

        input = self.construct_input_tensor(input_sequence)  # x
        target = self.construct_target_tensor(input_sequence)  # y
        padding_mask = self.construct_padding_mask(input)
        future_mask = self.construct_future_mask(input.size(0))
        return input, target, padding_mask, future_mask


class _MISSING_TYPE:
    """
    A sentinel class used to indicate that a parameter was not supplied.

    This is used to differentiate between cases where a parameter is not
    provided and where a parameter is provided with the value None. The class
    provides a more descriptive representation than None or other placeholders.

    NOTE: example usage is if you want to assign a default empty list or dict
    but it is mutable, so you assign this type but not None since None don't make
    sense.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(_MISSING_TYPE, cls).__new__(cls)
        return cls._instance

    def __repr__(self):
        return "<MISSING>"


# avoid name clash with dataclasses
_MISSING = _MISSING_TYPE()


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    batch_first: bool = True,
    pad_token_id: int = _MISSING,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    - **batch_first Parameter**: If `batch_first=True`, the resulting tensor
    `inputs_padded` will have a shape of `(batch_size, max_seq_len)`, where
    `batch_size` is the number of samples in the batch and `max_seq_len` is the
    length of the longest sequence in the batch. If `batch_first=False`, the shape
    will be `(max_seq_len, batch_size)`.

    - **padding_value Parameter**: The `padding_value` parameter specifies the value
    to use for padding shorter sequences. `pad_token_id` is typically used here,
    which should correspond to the padding token's ID in your vocabulary.
    """
    # omega confused during zipping so put here for clarity
    # that when you unzip a batch it becomes a tuple.
    inputs: Tuple[torch.Tensor, ...]
    targets: Tuple[torch.Tensor, ...]
    padding_masks: Tuple[torch.Tensor, ...]
    future_masks: Tuple[torch.Tensor, ...]

    inputs, targets, padding_masks, future_masks = zip(*batch)

    # Padding sequences to the same length
    inputs_padded: torch.Tensor = torch.nn.utils.rnn.pad_sequence(
        inputs, batch_first=batch_first, padding_value=pad_token_id
    )
    targets_padded: torch.Tensor = torch.nn.utils.rnn.pad_sequence(
        targets, batch_first=batch_first, padding_value=pad_token_id
    )
    padding_masks_padded: torch.Tensor = torch.nn.utils.rnn.pad_sequence(
        padding_masks, batch_first=True, padding_value=0
    )

    # Reshaping padding masks
    batch_size, seq_len = inputs_padded.size(0), inputs_padded.size(1)

    # padding_masks before view has shape: (batch_size, seq_len)
    # we want it to be (B, L, L) then (B, 1, L, L)
    padding_masks_padded = padding_masks_padded.view(batch_size, 1, 1, seq_len).expand(
        batch_size, 1, seq_len, seq_len
    )

    # future mask has shape (L, L) but we want it to be (B, L, L) then (B, 1, L, L)
    future_masks = torch.stack(future_masks)

    future_masks_expanded = future_masks.expand(batch_size, -1, -1).unsqueeze(1)
    # TODO: below is DEPRECATED see my notes for older explanation
    # future_masks_expanded = (
    #     future_masks.view(1, seq_len, seq_len)
    #     .expand(size=(batch_size, -1, -1))
    #     .unsqueeze(1)
    # )
    return inputs_padded, targets_padded, padding_masks_padded, future_masks_expanded


if __name__ == "__main__":
    vocab = AdderVocabulary.from_tokens(tokens=TOKENS)

    pprint(vocab.token_to_index)
    pprint(vocab.index_to_token)

    pprint(vocab.encode("1"))
    pprint(vocab.encode("+"))

    sequence = "15+57=072"
    sequences = ["15+57=072", "92+00=092", "95+53=148", "15+10=025"]

    encoded_sentence = vocab.encode(sequence)
    pprint(encoded_sentence)
    decoded_sentence = vocab.decode(encoded_sentence)
    pprint(decoded_sentence)

    encoded_sentences = vocab.encode_batch(sequences)
    pprint(encoded_sentences)
    decoded_sentences = vocab.decode_batch(encoded_sentences)
    pprint(decoded_sentences)

    dataset = AdderDataset(data=sequences, vocabulary=vocab)

    print()

    for x, y, pad_mask, future_mask in dataset:
        print("x")
        pprint(x)
        print("y")
        pprint(y)
        print("pad")
        pprint(pad_mask)
        print("future")
        pprint(future_mask)

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
            padding_masks_padded,
            future_masks_expanded,
        ) = batch

        # Print shapes
        print(f"Batch {i+1}")
        print("Inputs Shape:", inputs_padded.shape)
        print("Targets Shape:", targets_padded.shape)
        print("Padding Masks Shape:", padding_masks_padded.shape)
        print("Future Masks Shape:", future_masks_expanded.shape)

        # Print values (consider printing only a part of each tensor for large datasets)
        print("Inputs Values:", inputs_padded)
        print("Targets Values:", targets_padded)
        print("Padding Masks Values:", padding_masks_padded)
        print("Future Masks Values:", future_masks_expanded)

        # Add a separator for readability between batches
        print("-" * 50)

        # Optionally, break after a few batches to avoid too much output
        if i >= 2:  # Change this number based on how many batches you want to inspect
            break

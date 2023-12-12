from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class GroundTruth:
    # test bad sequences like 01+02=03?
    num_digits: int = 2
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

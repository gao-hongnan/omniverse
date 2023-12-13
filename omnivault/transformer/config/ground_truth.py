from dataclasses import dataclass, field
from typing import Dict, List

import torch


@dataclass
class GroundTruth:
    token_to_index: Dict[str, int] = field(  # TODO: consider using Literal
        default_factory=lambda: {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "+": 10,
            "*": 11,
            "-": 12,
            "=": 13,
            "<BOS>": 14,
            "<EOS>": 15,
            "<PAD>": 16,
            "<UNK>": 17,
        }
    )
    index_to_token: Dict[int, str] = field(
        default_factory=lambda: {
            0: "0",
            1: "1",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "+",
            11: "*",
            12: "-",
            13: "=",
            14: "<BOS>",
            15: "<EOS>",
            16: "<PAD>",
            17: "<UNK>",
        }
    )

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

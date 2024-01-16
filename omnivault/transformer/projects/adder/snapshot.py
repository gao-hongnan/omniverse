from dataclasses import dataclass, field
from typing import Dict, List

import torch

from omnivault.transformer.config.composer import DataConfig
from omnivault.transformer.config.constants import MaybeConstant
from omnivault.transformer.core.dataset import AdderDataset, AdderDatasetYield
from omnivault.transformer.core.tokenizer import AdderTokenizer
from omnivault.transformer.core.vocabulary import AdderVocabulary
from omnivault.transformer.utils.general_utils import download_and_read_sequences


@dataclass
class AdderGroundTruth:
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

    mock_batch: List[AdderDatasetYield] = field(init=False)

    inputs_collated: torch.LongTensor = field(
        default_factory=lambda: torch.LongTensor([[14, 1, 5, 10, 5, 7, 13, 0, 7, 2], [14, 0, 1, 10, 0, 2, 13, 0, 0, 3]])
    )
    targets_collated: torch.LongTensor = field(
        default_factory=lambda: torch.LongTensor(
            [
                [16, 16, 16, 16, 16, 16, 0, 7, 2, 15],
                [16, 16, 16, 16, 16, 16, 0, 0, 3, 15],
            ]
        )
    )

    padding_masks_collated: torch.BoolTensor = field(
        default_factory=lambda: torch.BoolTensor(
            [
                [
                    [
                        [True, True, True, True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True, True, True, True],
                    ]
                ],
                [
                    [
                        [True, True, True, True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True, True, True, True],
                    ]
                ],
            ]
        )
    )
    # shape: (batch_size, 1, seq_len, seq_len)
    future_masks_collated: torch.BoolTensor = field(
        default_factory=lambda: torch.BoolTensor(
            [
                [
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
                ],
                [
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
                ],
            ]
        )
    )

    def __post_init__(self) -> None:
        # Construct mock_batch using the existing fields
        self.mock_batch = [
            (self.inputs[i], self.targets[i], self.padding_masks[i], self.future_masks[i])
            for i in range(len(self.inputs))
        ]


ADDER_GROUND_TRUTH = AdderGroundTruth()

adder_mock_batch_ = ADDER_GROUND_TRUTH.mock_batch

constants = MaybeConstant(
    NUM_DIGITS=2,
    TOKENS=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "*", "-", "=", "<BOS>", "<EOS>", "<PAD>", "<UNK>"],
)
adder_vocab_ = AdderVocabulary.from_tokens(tokens=constants.TOKENS, num_digits=constants.NUM_DIGITS)  # type: ignore[attr-defined]
adder_tokenizer_ = AdderTokenizer(vocabulary=adder_vocab_)

data = DataConfig(
    context_length=11,
    dataset_name="adder_dataset",
    dataset_size=10000,
    dataset_path="./data/adder/adder_dataset.txt",
    dataset_dir="./data/adder",
    dataset_url="https://raw.githubusercontent.com/gao-hongnan/omniverse/dev/omnivault/transformer/projects/adder/assets/adder_dataset.txt",
    split=[0.7, 0.2, 0.1],
    collate_fn={"batch_first": True, "pad_token_id": 16},
    train_loader={"batch_size": 256, "shuffle": True, "num_workers": 0, "pin_memory": False, "drop_last": False},
    valid_loader={"batch_size": 256, "shuffle": False, "num_workers": 0, "pin_memory": False, "drop_last": False},
    test_loader={"batch_size": 128, "shuffle": False, "num_workers": 0, "pin_memory": False, "drop_last": False},
)


sequences = list(download_and_read_sequences(url=data.dataset_url, dataset_name=data.dataset_name))
adder_dataset_ = AdderDataset(
    data=sequences, tokenizer=adder_tokenizer_
)  # NOTE: for end2end test, so load a small dataset here.
adder_mock_dataset_ = AdderDataset(data=ADDER_GROUND_TRUTH.sequences, tokenizer=adder_tokenizer_)  # type: ignore[arg-type]

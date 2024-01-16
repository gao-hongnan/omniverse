from __future__ import annotations

from typing import List

import pytest

from omnivault.transformer.config.constants import MaybeConstant
from omnivault.transformer.core.dataset import AdderDataset, AdderDatasetYield
from omnivault.transformer.core.tokenizer import AdderTokenizer
from omnivault.transformer.core.vocabulary import AdderVocabulary
from omnivault.transformer.projects.adder.snapshot import (
    AdderGroundTruth,
    adder_mock_batch_,
    adder_mock_dataset_,
    adder_tokenizer_,
    adder_vocab_,
)


@pytest.fixture(scope="module")
def adder_ground_truth() -> AdderGroundTruth:
    """Define fixture so that we can use it in other tests with re-invoking the AdderGroundTruth class."""
    return AdderGroundTruth()


@pytest.fixture(scope="module")
def adder_mock_batch() -> List[AdderDatasetYield]:
    return adder_mock_batch_


@pytest.fixture(scope="module")
def maybe_constant() -> MaybeConstant:
    return MaybeConstant()


@pytest.fixture(scope="module")
def adder_vocab() -> AdderVocabulary:
    return adder_vocab_


@pytest.fixture(scope="module")
def adder_tokenizer() -> AdderTokenizer:
    return adder_tokenizer_


@pytest.fixture(scope="module")
def adder_dataset() -> AdderDataset:
    return adder_mock_dataset_


@pytest.fixture(scope="module")
def adder_dataset_but_larger(adder_tokenizer: AdderTokenizer) -> AdderDataset:
    sequences = AdderGroundTruth().sequences
    sequences = sequences * 100
    dataset: AdderDataset = AdderDataset(data=sequences, tokenizer=adder_tokenizer)
    return dataset

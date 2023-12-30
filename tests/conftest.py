from __future__ import annotations

from typing import List

import pytest

from omnivault.transformer.config.constants import MaybeConstant
from omnivault.transformer.config.ground_truth import GroundTruth
from omnivault.transformer.core.dataset import AdderDataset, AdderDatasetYield
from omnivault.transformer.core.tokenizer import AdderTokenizer
from omnivault.transformer.core.vocabulary import AdderVocabulary, Vocabulary


@pytest.fixture(scope="module")
def ground_truth() -> GroundTruth:
    """Define fixture so that we can use it in other tests with re-invoking the GroundTruth class."""
    return GroundTruth()


@pytest.fixture(scope="module")
def mock_batch(ground_truth: GroundTruth) -> List[AdderDatasetYield]:
    return ground_truth.mock_batch


@pytest.fixture(scope="module")
def maybe_constant() -> MaybeConstant:
    return MaybeConstant()


@pytest.fixture(scope="module")
def adder_vocab() -> Vocabulary:
    return AdderVocabulary.from_tokens(tokens=MaybeConstant().TOKENS, num_digits=MaybeConstant().NUM_DIGITS)


@pytest.fixture(scope="module")
def adder_tokenizer(adder_vocab: AdderVocabulary) -> AdderTokenizer:
    return AdderTokenizer(vocabulary=adder_vocab)


@pytest.fixture(scope="module")
def adder_dataset(adder_tokenizer: AdderTokenizer) -> AdderDataset:
    sequences = GroundTruth().sequences
    dataset: AdderDataset = AdderDataset(data=sequences, tokenizer=adder_tokenizer)
    return dataset


@pytest.fixture(scope="module")
def adder_dataset_but_larger(adder_tokenizer: AdderTokenizer) -> AdderDataset:
    sequences = GroundTruth().sequences
    sequences = sequences * 100
    dataset: AdderDataset = AdderDataset(data=sequences, tokenizer=adder_tokenizer)
    return dataset

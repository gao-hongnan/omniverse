import pytest

from omnivault.transformer.config.constants import NUM_DIGITS, TOKENS
from omnivault.transformer.config.ground_truth import GroundTruth
from omnivault.transformer.core.dataset import AdderDataset, AdderDatasetYield
from omnivault.transformer.core.vocabulary import AdderVocabulary, Vocabulary


@pytest.fixture(scope="module")
def ground_truth() -> GroundTruth:
    """Define fixture so that we can use it in other tests with re-invoking the GroundTruth class."""
    return GroundTruth()


@pytest.fixture(scope="module")
def adder_vocab() -> Vocabulary:
    return AdderVocabulary.from_tokens(tokens=TOKENS, num_digits=NUM_DIGITS)


@pytest.fixture(scope="module")
def adder_dataset(adder_vocab: Vocabulary) -> AdderDataset[AdderDatasetYield]:
    sequences = GroundTruth().sequences
    dataset: AdderDataset[AdderDatasetYield] = AdderDataset(data=sequences, vocabulary=adder_vocab)
    return dataset

import pytest

from omnivault.transformers.config.constants import TOKENS
from omnivault.transformers.config.ground_truth import GroundTruth
from omnivault.transformers.core.vocabulary import AdderVocabulary, Vocabulary
from omnivault.transformers.core.dataset import AdderDataset, AdderDatasetYield


@pytest.fixture(scope="module")
def ground_truth() -> GroundTruth:
    """Define fixture so that we can use it in other tests with re-invoking the GroundTruth class."""
    return GroundTruth()

@pytest.fixture(scope="module")
def adder_vocab() -> Vocabulary:
    return AdderVocabulary.from_tokens(tokens=TOKENS, num_digits=GroundTruth().num_digits)

@pytest.fixture(scope="module")
def adder_dataset(adder_vocab: Vocabulary) -> AdderDataset[AdderDatasetYield]:
    sequences = GroundTruth().sequences
    dataset: AdderDataset[AdderDatasetYield] = AdderDataset(data=sequences, vocabulary=adder_vocab)
    return dataset

from omnivault.transformer.config.ground_truth import GroundTruth
from omnivault.transformer.core.vocabulary import AdderVocabulary

GROUND_TRUTH = GroundTruth()


def test_len_vocab(adder_vocab: AdderVocabulary) -> None:
    """Test that the vocabulary length is correct."""
    assert len(adder_vocab) == len(adder_vocab.TOKENS) == adder_vocab.vocab_size


def test_vocab_token_to_index(adder_vocab: AdderVocabulary) -> None:
    """Test that the token to index mapping is correct."""
    assert adder_vocab.token_to_index == GROUND_TRUTH.token_to_index


def test_vocab_index_to_token(adder_vocab: AdderVocabulary) -> None:
    """Test that the index to token mapping is correct."""
    assert adder_vocab.index_to_token == GROUND_TRUTH.index_to_token

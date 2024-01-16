from omnivault.transformer.core.vocabulary import AdderVocabulary
from omnivault.transformer.projects.adder.snapshot import ADDER_GROUND_TRUTH, constants


def test_adder_vocab_len_same_as_num_of_tokens(adder_vocab: AdderVocabulary) -> None:
    """Test that the vocabulary length is correct."""
    assert len(adder_vocab) == len(adder_vocab.TOKENS) == adder_vocab.vocab_size


def test_adder_vocab_token_to_index(adder_vocab: AdderVocabulary) -> None:
    """Test that the token to index mapping is correct."""
    assert adder_vocab.token_to_index == ADDER_GROUND_TRUTH.token_to_index


def test_adder_vocab_index_to_token(adder_vocab: AdderVocabulary) -> None:
    """Test that the index to token mapping is correct."""
    assert adder_vocab.index_to_token == ADDER_GROUND_TRUTH.index_to_token


def test_adder_vocab_from_tokens(adder_vocab: AdderVocabulary) -> None:
    """Test that the from_tokens method correctly creates the mappings."""
    vocab = AdderVocabulary.from_tokens(constants.TOKENS)  # type: ignore[attr-defined]
    assert vocab.token_to_index == adder_vocab.token_to_index
    assert vocab.index_to_token == adder_vocab.index_to_token
    assert isinstance(vocab, type(adder_vocab))


def test_adder_vocab_num_digits(adder_vocab: AdderVocabulary) -> None:
    """Test that the num_digits attribute is correctly set."""
    assert adder_vocab.num_digits == 2


def test_adder_vocab_all_tokens_in_mappings(adder_vocab: AdderVocabulary) -> None:
    """Test that all tokens in the TOKENS list are in the mappings."""
    assert all(token in adder_vocab.token_to_index for token in adder_vocab.TOKENS)
    assert all(token in adder_vocab.index_to_token.values() for token in adder_vocab.TOKENS)

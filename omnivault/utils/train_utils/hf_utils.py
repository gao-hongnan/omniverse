from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, PreTrainedTokenizerFast


def maybe_resize_token_embeddings(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase | PreTrainedTokenizerFast
) -> bool:
    """Returns True if the token embeddings need to be resized to match the tokenizer's vocabulary size.

    Parameters
    ----------
    model : PreTrainedModel
        The model to check for token embeddings.
    tokenizer : PreTrainedTokenizerBase | PreTrainedTokenizerFast
        The tokenizer to check for vocabulary size.

    Returns
    -------
    bool
        Returns True if the token embeddings need to be resized, False otherwise.
    """
    try:
        embedding_module: torch.nn.Module = model.get_input_embeddings()
        embedding_size: int = embedding_module.weight.shape[0]
    except AttributeError as exc:
        raise AttributeError("`weight` attribute not found in the model's input embeddings. ") from exc
    tokenizer_vocab_size = len(tokenizer)
    return embedding_size != tokenizer_vocab_size

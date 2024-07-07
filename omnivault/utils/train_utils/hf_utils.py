from __future__ import annotations

from typing import Dict

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


def smart_tokenizer_and_embedding_resize(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase | PreTrainedTokenizerFast,
    special_tokens_dict: Dict[str, int],
) -> None:
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size
    not be divisible by 64. This is also referenced from `qlora/qlora.py`
    https://github.com/artidoro/qlora/blob/7f4e95a68dc076bea9b3a413d2b512eca6d004e5/qlora.py#L425.

    Parameters
    ----------
    model : PreTrainedModel
        The model to check for token embeddings.
    tokenizer : PreTrainedTokenizerBase | PreTrainedTokenizerFast
        The tokenizer to check for vocabulary size.
    special_tokens_dict : Dict[str, int]
        Dictionary containing special tokens to add.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg

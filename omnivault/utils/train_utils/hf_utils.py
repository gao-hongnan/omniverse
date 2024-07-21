from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from transformers import PreTrainedModel, PreTrainedTokenizerBase, PreTrainedTokenizerFast
from transformers.trainer_utils import EvalPrediction


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


def compute_metrics_for_single_label_classification(eval_prediction: EvalPrediction) -> Dict[str, float | List[float]]:
    logits, labels = eval_prediction.predictions, eval_prediction.label_ids
    probs = softmax(logits, axis=-1)

    num_classes = logits.shape[1]
    preds = np.argmax(probs, axis=1)

    metrics = {
        "eval_log_loss": log_loss(labels, probs),
        "eval_accuracy": accuracy_score(labels, preds),
        "eval_precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "eval_recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "eval_f1_score_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "eval_precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
        "eval_recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
        "eval_f1_score_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "eval_confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "eval_roc_auc": roc_auc_score(labels, probs, multi_class="ovr"),
        "eval_pr_auc": average_precision_score(labels, probs, average="macro"),
    }

    if num_classes == 2:
        metrics["eval_brier_score"] = brier_score_loss(labels, probs[:, 1], pos_label=1)
    else:
        brier_scores = [brier_score_loss(labels == i, probs[:, i]) for i in range(num_classes)]
        metrics["eval_brier_score"] = np.mean(brier_scores)

    if num_classes > 2:
        for class_index in range(num_classes):
            fpr, tpr, _ = roc_curve(labels == class_index, probs[:, class_index])
            roc_auc = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(labels == class_index, probs[:, class_index])
            pr_auc = auc(recall, precision)
            metrics[f"eval_roc_auc_class_{class_index}"] = roc_auc
            metrics[f"eval_pr_auc_class_{class_index}"] = pr_auc

    return metrics

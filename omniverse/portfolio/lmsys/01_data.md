# Data

## Design Patterns

### Factory

We first modularize our code to have a factory design so that we can easily add
new preprocessing modules for different LLMs. This is because different LLMs may
have vastly different template formats.

```python
from typing import TypedDict

class DataModule(TypedDict):
    tokenized_train_dataset: Dataset
    tokenized_valid_dataset: Dataset
    data_collator: DataCollator

def data_module_factory(
    composer: Composer,
    tokenizer: PreTrainedTokenizerBase,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    debug: bool = False,
) -> DataModule:
    return data_module_based_on_model
```

### Registry

We also can use a registry to store the data modules so that we can easily
access them later.

### Config Driven - Context Object Pattern

```python
class DataConfig(BaseModel):
    """Base class for dataset configuration."""

class TokenizerConfig(BaseModel):
    """Base class for tokenizer configuration."""
```

## Classification vs Generation

We mostly use the LLMs as a sequence classification model, we did some basic
statistics check on the dataset's class distribution to see if there's any
skewed distribution.

## Prompt Engineering

For example, if we use Llama 3, we need to use its special prompt format. A
sample prompt is given below:

```python
def make_llama_3_prompt(
    user_prompt: str,
    response_a: str,
    response_b: str,
    system_prompt: str | None = None,
) -> str:
    if system_prompt is None:
        system_prompt = """
        You are an AI trained to evaluate two responses returned by two other
        AI models to a user query.
        """

    _system_prompt = f"""
    <|start_header_id|>system<|end_header_id|>

    {system_prompt}<|eot_id|>
    """
    full_user_prompt = f"""
    Here is the user query: {user_prompt}

    Response from Model A:

    {response_a}

    Response from Model B:

    {response_b}
    """

    return f"""
    <|begin_of_text|>{_system_prompt}<|start_header_id|>user<|end_header_id|>

    {full_user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
```

Note very carefully that this prompt template is for text classification and not
text generation. If for example, we are using Llama 3 for text generation, we
can write the prompt like:

```text
Given the following user query: {user_prompt}

Response from Model A: {response_a}

Response from Model B: {response_b}

Please act as a judge and say `win_a` if you think response A is better than
response B, `win_b` if you think response B is better than response A, or `tie`
if you think both responses are equally good.

Answer: {ground_truth_label}
```

## Tokenization

For any model family, we need to use the same tokenizer that was used to train
the model for consistency.

```python
class TokenizerConfig(BaseModel):
    """Base class for tokenizer configuration."""

    # from_pretrained(...) args
    padding_side: Literal["left", "right"] = "left"

    # tokenizer(...) args
    max_length: int = 1024  # alias context_window/context_length
    truncation: Literal["do_not_truncate", "longest_first", "only_first", "only_second"] | bool = True
    return_tensors: Literal["pt", "tf", "np"] | None = None
    add_special_tokens: bool = False
    padding: Literal["longest", "max_length", "do_not_pad"] | bool = "longest"
    return_attention_mask: bool = False
```

## Resampling And Splitting

Ensure no data leakage between train and validation set. Sometimes if there's
apparent groups in the data, we need to stratify the data sampling so that the
same group is not split between train and validation set. You can use
`StratifiedGroupKFold` from `sklearn.model_selection` to achieve this.

-   Stratified Sampling
-   Stratified Group Sampling

Sample code seen below:

```python
from __future__ import annotations

from typing import Any, Dict, Literal

import pandas as pd
from sklearn import model_selection
from sklearn.model_selection._split import BaseCrossValidator


def create_folds(
    df: pd.DataFrame,
    *,
    resample_strategy: str,
    resample_params: Dict[str, Any],
    group_by: str | None = None,
    stratify_by: str | None = None,
    fold_column: Literal["fold"] = "fold",
) -> pd.DataFrame:
    """
    Assign fold numbers to rows in a DataFrame based on specified resampling
    strategy.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to which the folds will be assigned.
    resample_strategy : str
        The resampling strategy, corresponds to sklearn's model_selection methods.
    resample_params : dict
        Parameters to pass to the resampling strategy constructor.
    group_by : str or None, optional
        Column to group data before splitting, by default None.
    stratify_by : str or None, optional
        Column to stratify split, by default None.
    fold_column : str, optional
        Name of the column to store fold numbers, by default "fold".

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional column indicating fold numbers.

    Notes
    -----
    Omit the use of `train_test_split` since the same result can be achieved by
    using `(Stratified)(Group)KFold` with `n_splits=2`.
    """
    cv: BaseCrossValidator = getattr(model_selection, resample_strategy)(**resample_params)
    stratify = df[stratify_by].values if stratify_by else None
    groups = df[group_by].values if group_by else None

    for _fold, (_train_idx, valid_idx) in enumerate(cv.split(df, stratify, groups)):
        df.loc[valid_idx, fold_column] = _fold
    df[fold_column] = df[fold_column].astype(int)
    return df
```

## Debugging

We have debug mode where we only take a subset of the dataset to run the
experiment. This is useful for debugging and for faster experimentation.

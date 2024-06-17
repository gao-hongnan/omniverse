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

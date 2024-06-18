from __future__ import annotations

import gc
from typing import List

import torch


def purge_global_scope(variable_name_or_names: str | List[str]) -> None:
    """
    Deletes the provided objects and performs cleanup.

    Parameters
    ----------
    objects: List[Any]
        The list of objects to be deleted.

    Notes
    -----
    The below does not work because we are passing in references of the
    variables and are not really deleted.

    ```python
    def purge_global_scope(object_or_objects: Any | List[Any]) -> None:
        if isinstance(object_or_objects, list):
            for obj in object_or_objects:
                del obj
        else:
            del object_or_objects

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    ```
    """
    if isinstance(variable_name_or_names, str):
        variable_name_or_names = [variable_name_or_names]

    global_vars = globals()
    for name in variable_name_or_names:
        if name in global_vars:
            del global_vars[name]

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

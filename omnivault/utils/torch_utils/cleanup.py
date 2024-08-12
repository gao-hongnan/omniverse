from __future__ import annotations

import gc
import sys
from typing import List

import torch


# FIXME: this does not work actually because local scope is not deleted.
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

    Secondly, this code is now working if I call it from another script, if you
    replace `caller_globals` with `globals()` it won't work if this function
    is called from another script.
    """
    if isinstance(variable_name_or_names, str):
        variable_name_or_names = [variable_name_or_names]

    caller_frame = sys._getframe(1)
    caller_locals = caller_frame.f_locals
    caller_globals = caller_frame.f_globals

    for name in variable_name_or_names:
        if name in caller_locals:
            del caller_locals[name]
        elif name in caller_globals:
            del caller_globals[name]

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

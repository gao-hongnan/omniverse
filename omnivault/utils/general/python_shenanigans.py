import inspect
from inspect import Signature
from typing import Any, Dict, Type, get_type_hints


def get_init_args(cls: Type[Any]) -> Dict[str, Any]:
    """
    Get the initialization arguments and their default values for a class.
    This does not play well with very complex classes such as Pydantic,
    you can just use the `model_fields` to get.

    Parameters
    ----------
    cls: Type[Any]
        The class to get the initialization arguments for.

    Returns
    -------
    Dict[str, Any]
        A dictionary where the keys are the argument names and the values are
        the default values. If an argument does not have a default value,
        its value in the dictionary will be None.
    """
    sig: Signature = inspect.signature(cls.__init__)
    type_hints: Dict[str, Any] = get_type_hints(cls.__init__)
    return {
        name: (param.default if param.default is not param.empty else None, type_hints.get(name))
        for name, param in sig.parameters.items()
        if name != "self"
    }

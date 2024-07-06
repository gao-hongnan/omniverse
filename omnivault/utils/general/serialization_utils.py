import json
import pickle
from typing import Any


def dump_json(data: Any, filepath: str, **kwargs: Any) -> None:
    """
    Dumps data to a JSON file.

    Parameters
    ----------
    data : Any
        The Python data to dump to the JSON file.
    filepath : str
        The path to the JSON file to write to.
    kwargs: Any
        Additional keyword arguments for `json.dump()`.
    """
    with open(filepath, "w") as file:
        json.dump(data, file, **kwargs)


def load_json(filepath: str, **kwargs: Any) -> Any:
    """
    Loads data from a JSON file.

    Parameters
    ----------
    filepath : str
        The path to the JSON file to read from.
    kwargs: Any
        Additional keyword arguments for `json.load()`.

    Returns
    -------
    Any
        The Python data loaded from the JSON file.
    """
    with open(filepath, "r") as file:
        return json.load(file, **kwargs)


def pickle_dump(obj: Any, filepath: str) -> None:
    """
    Serializes an object to a binary file using pickle.

    Parameters
    ----------
    obj : Any
        The object to serialize.
    filepath : str
        The path to the file where the object will be stored.
    """
    with open(filepath, "wb") as file:
        pickle.dump(obj, file)


def pickle_load(filepath: str) -> Any:
    """
    Deserializes an object from a binary file using pickle.

    Parameters
    ----------
    filepath : str
        The path to the file from which the object will be loaded.

    Returns
    -------
    Any
        The deserialized Python object.
    """
    with open(filepath, "rb") as file:
        return pickle.load(file)

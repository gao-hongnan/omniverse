import json
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
    with open(filepath, 'w') as file:
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
    with open(filepath, 'r') as file:
        return json.load(file, **kwargs)


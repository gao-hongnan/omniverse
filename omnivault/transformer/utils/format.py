from __future__ import annotations

from typing import Dict, List


def format_lr(lr_or_lrs: float | List[float], precision: int) -> str:
    format_str = f"%.{precision}f"
    if isinstance(lr_or_lrs, list):
        return ", ".join([format_str % lr for lr in lr_or_lrs])
    return format_str % lr_or_lrs


def create_markdown_table(data: Dict[str, List[int | float]]) -> str:
    """Create a markdown table from a dictionary of lists. Run through prettier
    for auto markdown formatting.

    Example
    -------
    >>> data = {
    ...     "Epoch": list(range(1, 4)),
    ...     "Train Avg Loss": [2.4211656037739346, 1.3809500090735298, 1.0856563610349383],
    ...     "Train Avg Perplexity": [11.258975982666016, 3.9786794185638428, 2.961383104324341],
    ...     "Valid Avg Loss": [1.7226673784255981, 1.1581441555023193, 1.000551365852356],
    ...     "Valid Avg Perplexity": [5.599444389343262, 3.184018611907959, 2.719780921936035]
    ... }
    >>> print(create_markdown_table(data))
    """
    headers = " | ".join(data.keys())
    markdown_table = f"| {headers} |\n"

    separator = " | ".join(["-----"] * len(data))
    markdown_table += f"| {separator} |\n"

    for i in range(len(next(iter(data.values())))):
        row = " | ".join(f"{values[i]:.0f}" if key == "Epoch" else f"{values[i]:.8f}" for key, values in data.items())
        markdown_table += f"| {row} |\n"

    return markdown_table
